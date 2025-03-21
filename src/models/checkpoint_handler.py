"""
Checkpoint handling utilities for LoRA fine-tuning.
Manages checkpoint saving, loading, and rotation.
"""

import os
import glob
import json
import shutil
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class CheckpointHandler:
    """
    Class for managing checkpoints during fine-tuning.
    Supports saving, loading, and rotating checkpoints.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 3,
        save_optimizer_state: bool = True,
        checkpoint_prefix: str = "checkpoint"
    ):
        """
        Initialize the checkpoint handler.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_optimizer_state: Whether to save optimizer state in checkpoints
            checkpoint_prefix: Prefix for checkpoint directory names
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.save_optimizer_state = save_optimizer_state
        self.checkpoint_prefix = checkpoint_prefix
        
        # Create the checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize metadata
        self.metadata_file = os.path.join(checkpoint_dir, "checkpoint_metadata.json")
        self._initialize_metadata()
        
        logger.info(f"CheckpointHandler initialized at {checkpoint_dir}")
    
    def _initialize_metadata(self) -> None:
        """Initialize or load checkpoint metadata file."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded checkpoint metadata with {len(self.metadata.get('checkpoints', []))} checkpoints")
        else:
            self.metadata = {
                "last_checkpoint": None,
                "checkpoints": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            self._save_metadata()
            logger.info("Initialized new checkpoint metadata")
    
    def _save_metadata(self) -> None:
        """Save checkpoint metadata to file."""
        self.metadata["updated_at"] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _get_checkpoint_path(self, checkpoint_id: str) -> str:
        """
        Get the path for a specific checkpoint.
        
        Args:
            checkpoint_id: Checkpoint identifier
            
        Returns:
            Path to the checkpoint directory
        """
        return os.path.join(self.checkpoint_dir, f"{self.checkpoint_prefix}-{checkpoint_id}")
    
    def _generate_checkpoint_id(self) -> str:
        """
        Generate a unique checkpoint ID.
        
        Returns:
            Checkpoint identifier
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{timestamp}"
    
    def save_checkpoint(
        self,
        model,
        optimizer=None,
        scheduler=None,
        step: int = 0,
        epoch: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save (optional)
            scheduler: Learning rate scheduler to save (optional)
            step: Current training step
            epoch: Current training epoch
            metrics: Training metrics (optional)
            extra_data: Additional data to save (optional)
            
        Returns:
            Checkpoint ID
        """
        # Generate checkpoint ID and path
        checkpoint_id = self._generate_checkpoint_id()
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        
        logger.info(f"Saving checkpoint {checkpoint_id} to {checkpoint_path}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model state
        model.save_pretrained(checkpoint_path)
        
        # Create checkpoint state file
        checkpoint_state = {
            "step": step,
            "epoch": epoch,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics or {},
            "extra_data": extra_data or {}
        }
        
        # Save optimizer state if requested
        if optimizer is not None and self.save_optimizer_state:
            optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
            torch_save_path = os.path.join(checkpoint_path, "training_state.pt")
            
            import torch
            torch.save(optimizer.state_dict(), optimizer_path)
            
            # Save training state (includes optimizer, scheduler, rng states)
            training_state = {
                "optimizer": optimizer.state_dict(),
                "step": step,
                "epoch": epoch,
            }
            
            # Add scheduler if provided
            if scheduler is not None:
                training_state["scheduler"] = scheduler.state_dict()
                
            # Add RNG states
            training_state["random_states"] = {
                "python": None,  # Python state isn't serializable directly
                "numpy": None,   # Will be a bytestring
                "torch": torch.get_rng_state(),
                "cuda": [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None
            }
            
            # Save random states separately
            import pickle
            import numpy as np
            import random
            
            try:
                # Save Python RNG state
                with open(os.path.join(checkpoint_path, "python_rng.pkl"), 'wb') as f:
                    pickle.dump(random.getstate(), f)
                
                # Save NumPy RNG state
                np.save(os.path.join(checkpoint_path, "numpy_rng.npy"), np.random.get_state())
                
                checkpoint_state["has_rng_states"] = True
            except Exception as e:
                logger.warning(f"Failed to save RNG states: {e}")
                checkpoint_state["has_rng_states"] = False
            
            # Save training state
            torch.save(training_state, torch_save_path)
            checkpoint_state["has_training_state"] = True
        else:
            checkpoint_state["has_training_state"] = False
        
        # Save checkpoint state
        with open(os.path.join(checkpoint_path, "checkpoint_state.json"), 'w') as f:
            json.dump(checkpoint_state, f, indent=2)
        
        # Update metadata
        checkpoint_info = {
            "id": checkpoint_id,
            "path": checkpoint_path,
            "step": step,
            "epoch": epoch,
            "created_at": checkpoint_state["created_at"],
            "metrics": metrics or {}
        }
        
        self.metadata["checkpoints"].append(checkpoint_info)
        self.metadata["last_checkpoint"] = checkpoint_id
        self._save_metadata()
        
        # Rotate old checkpoints if necessary
        self._rotate_checkpoints()
        
        return checkpoint_id
    
    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints if the maximum number is exceeded."""
        checkpoints = self.metadata["checkpoints"]
        
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Sort checkpoints by creation time (oldest first)
        sorted_checkpoints = sorted(
            checkpoints, 
            key=lambda x: datetime.fromisoformat(x["created_at"])
        )
        
        # Remove oldest checkpoints
        checkpoints_to_remove = sorted_checkpoints[:len(sorted_checkpoints) - self.max_checkpoints]
        
        for checkpoint in checkpoints_to_remove:
            checkpoint_id = checkpoint["id"]
            checkpoint_path = checkpoint["path"]
            
            logger.info(f"Rotating out checkpoint {checkpoint_id}")
            
            # Remove directory
            if os.path.exists(checkpoint_path):
                shutil.rmtree(checkpoint_path)
            
            # Update metadata
            self.metadata["checkpoints"] = [
                c for c in self.metadata["checkpoints"] if c["id"] != checkpoint_id
            ]
        
        self._save_metadata()
    
    def load_latest_checkpoint(self, model=None, optimizer=None, scheduler=None):
        """
        Load the latest checkpoint.
        
        Args:
            model: Model to load checkpoint into (optional)
            optimizer: Optimizer to load state into (optional)
            scheduler: Learning rate scheduler to load state into (optional)
            
        Returns:
            Tuple of (model, optimizer, scheduler, checkpoint_state)
        """
        # Get latest checkpoint ID
        latest_checkpoint_id = self.metadata.get("last_checkpoint")
        
        if latest_checkpoint_id is None:
            logger.warning("No checkpoints found to load")
            return model, optimizer, scheduler, None
        
        return self.load_checkpoint(latest_checkpoint_id, model, optimizer, scheduler)
    
    def load_checkpoint(
        self, 
        checkpoint_id: str, 
        model=None, 
        optimizer=None, 
        scheduler=None
    ):
        """
        Load a specific checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID to load
            model: Model to load checkpoint into (optional)
            optimizer: Optimizer to load state into (optional)
            scheduler: Learning rate scheduler to load state into (optional)
            
        Returns:
            Tuple of (model, optimizer, scheduler, checkpoint_state)
        """
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint {checkpoint_id} not found at {checkpoint_path}")
            return model, optimizer, scheduler, None
        
        logger.info(f"Loading checkpoint {checkpoint_id} from {checkpoint_path}")
        
        # Load checkpoint state
        checkpoint_state_path = os.path.join(checkpoint_path, "checkpoint_state.json")
        if os.path.exists(checkpoint_state_path):
            with open(checkpoint_state_path, 'r') as f:
                checkpoint_state = json.load(f)
        else:
            logger.warning(f"Checkpoint state file not found at {checkpoint_state_path}")
            checkpoint_state = {}
        
        # Load model if provided
        if model is not None:
            model.load_adapter_from_pretrained(checkpoint_path)
            logger.info(f"Loaded model weights from checkpoint {checkpoint_id}")
        
        # Load training state if available
        if checkpoint_state.get("has_training_state", False):
            import torch
            training_state_path = os.path.join(checkpoint_path, "training_state.pt")
            
            if os.path.exists(training_state_path):
                try:
                    training_state = torch.load(training_state_path, map_location="cpu")
                    
                    # Load optimizer state if provided
                    if optimizer is not None and "optimizer" in training_state:
                        optimizer.load_state_dict(training_state["optimizer"])
                        logger.info(f"Loaded optimizer state from checkpoint {checkpoint_id}")
                    
                    # Load scheduler state if provided
                    if scheduler is not None and "scheduler" in training_state:
                        scheduler.load_state_dict(training_state["scheduler"])
                        logger.info(f"Loaded scheduler state from checkpoint {checkpoint_id}")
                    
                    # Load RNG states if available
                    if "random_states" in training_state and checkpoint_state.get("has_rng_states", False):
                        import pickle
                        import numpy as np
                        import random
                        
                        # Load Python RNG state
                        python_rng_path = os.path.join(checkpoint_path, "python_rng.pkl")
                        if os.path.exists(python_rng_path):
                            with open(python_rng_path, 'rb') as f:
                                random.setstate(pickle.load(f))
                        
                        # Load NumPy RNG state
                        numpy_rng_path = os.path.join(checkpoint_path, "numpy_rng.npy")
                        if os.path.exists(numpy_rng_path):
                            np_state = np.load(numpy_rng_path, allow_pickle=True)
                            np.random.set_state(np_state)
                        
                        # Load torch RNG states
                        if "torch" in training_state["random_states"]:
                            torch.set_rng_state(training_state["random_states"]["torch"])
                        
                        # Load CUDA RNG states
                        cuda_states = training_state["random_states"].get("cuda")
                        if cuda_states is not None and torch.cuda.is_available():
                            for i, state in enumerate(cuda_states):
                                if i < torch.cuda.device_count():
                                    torch.cuda.set_rng_state(state, i)
                        
                        logger.info(f"Restored random states from checkpoint {checkpoint_id}")
                
                except Exception as e:
                    logger.error(f"Error loading training state: {e}")
            else:
                logger.warning(f"Training state file not found at {training_state_path}")
        
        return model, optimizer, scheduler, checkpoint_state
    
    def get_all_checkpoints(self) -> List[Dict[str, Any]]:
        """
        Get information about all available checkpoints.
        
        Returns:
            List of checkpoint information dictionaries
        """
        return self.metadata.get("checkpoints", [])
    
    def find_checkpoint_by_step(self, step: int) -> Optional[str]:
        """
        Find a checkpoint by training step.
        
        Args:
            step: Training step to find
            
        Returns:
            Checkpoint ID if found, None otherwise
        """
        for checkpoint in self.metadata.get("checkpoints", []):
            if checkpoint.get("step") == step:
                return checkpoint.get("id")
        return None
    
    def find_checkpoint_by_epoch(self, epoch: int) -> Optional[str]:
        """
        Find a checkpoint by training epoch.
        
        Args:
            epoch: Training epoch to find
            
        Returns:
            Checkpoint ID if found, None otherwise
        """
        for checkpoint in self.metadata.get("checkpoints", []):
            if checkpoint.get("epoch") == epoch:
                return checkpoint.get("id")
        return None
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a specific checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint {checkpoint_id} not found at {checkpoint_path}")
            return False
        
        logger.info(f"Deleting checkpoint {checkpoint_id}")
        
        # Remove directory
        shutil.rmtree(checkpoint_path)
        
        # Update metadata
        self.metadata["checkpoints"] = [
            c for c in self.metadata["checkpoints"] if c["id"] != checkpoint_id
        ]
        
        # Update last_checkpoint if necessary
        if self.metadata.get("last_checkpoint") == checkpoint_id:
            remaining_checkpoints = self.metadata["checkpoints"]
            if remaining_checkpoints:
                # Sort by creation time and get the most recent
                sorted_checkpoints = sorted(
                    remaining_checkpoints,
                    key=lambda x: datetime.fromisoformat(x["created_at"]),
                    reverse=True
                )
                self.metadata["last_checkpoint"] = sorted_checkpoints[0]["id"]
            else:
                self.metadata["last_checkpoint"] = None
        
        self._save_metadata()
        return True
