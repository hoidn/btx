# btx/processing/core/adapters.py
"""Task adaptation and registry components."""

from typing import Dict, Any, Type, Optional, TypeVar, Generic, Callable
from dataclasses import dataclass
import inspect
from datetime import datetime
from pathlib import Path

from .task import Task

# Generic type variables for input/output types
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')

class AdapterValidationError(Exception):
    """Raised when task adapter validation fails."""
    pass

@dataclass
class TaskAdapter(Task, Generic[InputT, OutputT]):
    """Adapter to wrap existing task classes into pipeline-compatible tasks.
    
    This adapter allows existing task classes to be used in pipelines without
    modification by wrapping them in a Task-compatible interface.
    
    Type Parameters:
        InputT: Type of task input
        OutputT: Type of task output
    """
    
    task_instance: Any  # The actual task instance
    input_type: Type[InputT]
    output_type: Type[OutputT]
    name: str  # Added for better error messaging
    
    def __init__(
        self,
        task_class: Type[Any],
        config: Dict[str, Any],
        input_type: Type[InputT],
        output_type: Type[OutputT],
        name: str
    ):
        """Initialize task adapter.
        
        Args:
            task_class: Original task class to wrap
            config: Configuration dictionary
            input_type: Expected input type (e.g., LoadDataInput)
            output_type: Expected output type (e.g., LoadDataOutput)
            name: Task name for error reporting
            
        Raises:
            AdapterValidationError: If task interface validation fails
        """
        super().__init__(config)
        self.task_instance = task_class(config)
        self.input_type = input_type
        self.output_type = output_type
        self.name = name
        
        # Validate task interface
        self._validate_task_interface()
    
    def _validate_config(self) -> None:
        """Config validation is handled by wrapped task's __init__."""
        pass
    
    def _validate_task_interface(self) -> None:
        """Validate that wrapped task has required interface.
        
        Raises:
            AdapterValidationError: If validation fails
        """
        # Check for run method
        if not hasattr(self.task_instance, 'run'):
            raise AdapterValidationError(
                f"Task {self.name} ({type(self.task_instance).__name__}) "
                "must have a 'run' method"
            )
        
        # Validate run method signature
        sig = inspect.signature(self.task_instance.run)
        params = list(sig.parameters.values())
        
        # Must have self and input_data parameter
        if len(params) != 2:  # self and input_data
            raise AdapterValidationError(
                f"Task {self.name} run method must have exactly two parameters "
                f"(got {len(params)})"
            )
        
        # Second parameter (after self) must be input_data
        input_param = params[1]
        if input_param.name != 'input_data':
            raise AdapterValidationError(
                f"Task {self.name} run method second parameter must be named "
                "'input_data' (got '{input_param.name}')"
            )
        
        # Check for plot_diagnostics method
        if hasattr(self.task_instance, 'plot_diagnostics'):
            diag_sig = inspect.signature(self.task_instance.plot_diagnostics)
            diag_params = list(diag_sig.parameters.values())
            if len(diag_params) != 3:  # self, output, save_dir
                raise AdapterValidationError(
                    f"Task {self.name} plot_diagnostics method must have exactly "
                    f"three parameters (got {len(diag_params)})"
                )
            
            # Validate parameter names
            expected_names = ['self', 'output', 'save_dir']
            actual_names = [p.name for p in diag_params]
            if actual_names != expected_names:
                raise AdapterValidationError(
                    f"Task {self.name} plot_diagnostics method must have parameters "
                    f"named: {', '.join(expected_names)} (got: {', '.join(actual_names)})"
                )
    
    def run(self, input_data: Optional[InputT] = None) -> OutputT:
        """Run the wrapped task with type checking.
        
        Args:
            input_data: Input data for task
            
        Returns:
            Task output
            
        Raises:
            TypeError: If input or output types don't match expectations
            RuntimeError: If task execution fails
        """
        # Validate input type if provided
        if input_data is not None and not isinstance(input_data, self.input_type):
            raise TypeError(
                f"Task {self.name} expected input type {self.input_type.__name__}, "
                f"got {type(input_data).__name__}"
            )
        
        # Run wrapped task
        try:
            result = self.task_instance.run(input_data)
        except Exception as e:
            raise RuntimeError(
                f"Task {self.name} execution failed: {str(e)}"
            ) from e
        
        # Validate output type
        if not isinstance(result, self.output_type):
            raise TypeError(
                f"Task {self.name} returned type {type(result).__name__}, "
                f"expected {self.output_type.__name__}"
            )
        
        return result
    
    def plot_diagnostics(self, output: OutputT, save_dir: Optional[Path] = None) -> None:
        """Generate diagnostic plots if wrapped task supports it.
        
        Args:
            output: Task output data
            save_dir: Directory to save plots
        """
        if hasattr(self.task_instance, 'plot_diagnostics'):
            self.task_instance.plot_diagnostics(output, save_dir)

class TaskRegistry:
    """Registry of task adapters with their input/output types.
    
    This registry maintains a mapping of task names to their implementations
    and type specifications, allowing tasks to be created by name while
    maintaining type safety.
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._registry: Dict[str, tuple[Type[Any], Type[Any], Type[Any]]] = {}
    
    def register(
        self,
        name: str,
        task_class: Type[Any],
        input_type: Type[Any],
        output_type: Type[Any]
    ) -> None:
        """Register a task with its types.
        
        Args:
            name: Task name
            task_class: Task implementation class
            input_type: Expected input type
            output_type: Expected output type
            
        Raises:
            ValueError: If task name is already registered
        """
        if name in self._registry:
            raise ValueError(f"Task {name} is already registered")
        
        self._registry[name] = (task_class, input_type, output_type)
    
    def create(self, name: str, config: Dict[str, Any]) -> TaskAdapter:
        """Create an adapter instance for a registered task.
        
        Args:
            name: Task name
            config: Task configuration
            
        Returns:
            Configured TaskAdapter instance
            
        Raises:
            ValueError: If task is not registered
        """
        if name not in self._registry:
            raise ValueError(f"Task {name} not registered")
        
        task_class, input_type, output_type = self._registry[name]
        return TaskAdapter(
            task_class=task_class,
            config=config,
            input_type=input_type,
            output_type=output_type,
            name=name
        )
    
    def list_tasks(self) -> Dict[str, Dict[str, str]]:
        """Get information about registered tasks.
        
        Returns:
            Dictionary mapping task names to their type information
        """
        return {
            name: {
                'task_class': task_class.__name__,
                'input_type': input_type.__name__,
                'output_type': output_type.__name__
            }
            for name, (task_class, input_type, output_type) in self._registry.items()
        }

# Global registry instance
registry = TaskRegistry()

def register_task(
    name: str,
    input_type: Type[Any],
    output_type: Type[Any]
) -> Callable[[Type[Any]], Type[Any]]:
    """Decorator to register a task class with the global registry.
    
    Args:
        name: Task name
        input_type: Expected input type
        output_type: Expected output type
        
    Returns:
        Decorator function
        
    Example:
        @register_task("load_data", LoadDataInput, LoadDataOutput)
        class LoadData:
            ...
    """
    def decorator(task_class: Type[Any]) -> Type[Any]:
        registry.register(name, task_class, input_type, output_type)
        return task_class
    return decorator