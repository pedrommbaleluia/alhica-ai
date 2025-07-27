from fastapi import FastAPI
from importlib import import_module

app = FastAPI(title="Alhica AI Autonomous")

# mount existing core
core = import_module('alhica_ai_core')
app.include_router(core.router)
