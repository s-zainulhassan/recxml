from src.config import config, set_seed
from src.models.student import StudentGenerator
from src.pipeline import build_model

set_seed(42)
config.ensure_directories()

student = build_model(StudentGenerator(config), config)
student.save_weights(config.initial_student_checkpoint)

print("Initial student checkpoint created.")
