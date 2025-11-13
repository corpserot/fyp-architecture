Push-Location $PSScriptRoot
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. --mypy_out=. cvengine.proto
Pop-Location