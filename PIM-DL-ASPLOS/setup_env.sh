#!/usr/bin/env bash

# Usage:
#   source ./setup_env.sh
#   source ./setup_env.sh --configure
#   source ./setup_env.sh --build
#
# This script prepares the environment required by PIM-DL inference-engine
# builds on top of the UPMEM SDK + pyenv Python 3.10.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced so exported variables persist."
  echo "Use: source ${0}"
  exit 1
fi

DO_CONFIGURE=0
DO_BUILD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --configure)
      DO_CONFIGURE=1
      ;;
    --build)
      DO_CONFIGURE=1
      DO_BUILD=1
      ;;
    -h|--help)
      echo "Usage: source ./setup_env.sh [--configure] [--build]"
      return 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: source ./setup_env.sh [--configure] [--build]"
      return 1
      ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
INFERENCE_DIR="${PROJECT_DIR}/inference-engine"
BUILD_DIR="${INFERENCE_DIR}/build"

UPMEM_HOME_DEFAULT="${HOME}/master/upmem-2025.1.0-Linux-x86_64"
UPMEM_HOME="${UPMEM_HOME:-${UPMEM_HOME_DEFAULT}}"
PYENV_ROOT="${PYENV_ROOT:-${HOME}/.pyenv}"
PYENV_ENV_NAME="${PYENV_ENV_NAME:-pimdl-py310}"
PYENV_PYVER="${PYENV_PYVER:-3.10.16}"
PY310_LIB="${PYENV_ROOT}/versions/${PYENV_PYVER}/lib"

if [[ ! -d "${UPMEM_HOME}" ]]; then
  echo "UPMEM_HOME not found: ${UPMEM_HOME}"
  echo "Set UPMEM_HOME then retry, for example:"
  echo "  export UPMEM_HOME=${UPMEM_HOME_DEFAULT}"
  return 1
fi

if [[ ! -f "${UPMEM_HOME}/upmem_env.sh" ]]; then
  echo "Missing UPMEM environment script: ${UPMEM_HOME}/upmem_env.sh"
  return 1
fi

if [[ ! -d "${INFERENCE_DIR}" ]]; then
  echo "Missing inference-engine directory: ${INFERENCE_DIR}"
  return 1
fi

if ! command -v pyenv >/dev/null 2>&1; then
  export PATH="${PYENV_ROOT}/bin:${PATH}"
fi

if ! command -v pyenv >/dev/null 2>&1; then
  echo "pyenv is not available in PATH and could not be found in ${PYENV_ROOT}/bin."
  return 1
fi

# Load UPMEM defaults (UPMEM_HOME, PATH, LD_LIBRARY_PATH, PYTHONPATH, etc.).
source "${UPMEM_HOME}/upmem_env.sh" || {
  echo "Failed to source ${UPMEM_HOME}/upmem_env.sh"
  return 1
}

eval "$(pyenv init -)" || {
  echo "Failed to initialize pyenv."
  return 1
}

if ! pyenv versions --bare | rg -q "^${PYENV_ENV_NAME}\$"; then
  echo "pyenv env '${PYENV_ENV_NAME}' not found."
  echo "Create it with: pyenv virtualenv ${PYENV_PYVER} ${PYENV_ENV_NAME}"
  return 1
fi

pyenv shell "${PYENV_ENV_NAME}" || {
  echo "Failed to activate pyenv shell ${PYENV_ENV_NAME}."
  return 1
}

if [[ ! -f "${PY310_LIB}/libpython3.10.so.1.0" ]]; then
  echo "Missing Python shared library: ${PY310_LIB}/libpython3.10.so.1.0"
  echo "Reinstall Python ${PYENV_PYVER} with pyenv if needed."
  return 1
fi

# Ensure pkg-config can find dpu.pc.
export PKG_CONFIG_PATH="${UPMEM_HOME}/share/pkgconfig${PKG_CONFIG_PATH+:${PKG_CONFIG_PATH}}"

# Ensure linker and runtime loader can resolve libdpu.so and libpython3.10.so.1.0.
export LD_LIBRARY_PATH="${PY310_LIB}:${UPMEM_HOME}/lib${LD_LIBRARY_PATH+:${LD_LIBRARY_PATH}}"
export LIBRARY_PATH="${UPMEM_HOME}/lib:${PY310_LIB}${LIBRARY_PATH+:${LIBRARY_PATH}}"
export LDFLAGS="-Wl,-rpath,${UPMEM_HOME}/lib -Wl,-rpath,${PY310_LIB} ${LDFLAGS:-}"

# Keep Python package visibility consistent with the selected pyenv env.
export PATH="${PYENV_ROOT}/shims:${PATH}"

mkdir -p "${BUILD_DIR}" || {
  echo "Failed to create build directory: ${BUILD_DIR}"
  return 1
}

echo "Environment is ready."
echo "  UPMEM_HOME=${UPMEM_HOME}"
echo "  pyenv env=${PYENV_ENV_NAME}"
echo "  python=$(python --version 2>&1)"

if [[ ${DO_CONFIGURE} -eq 1 ]]; then
  echo "Configuring inference-engine..."
  cmake -S "${INFERENCE_DIR}" -B "${BUILD_DIR}"
fi

if [[ ${DO_BUILD} -eq 1 ]]; then
  echo "Building inference-engine..."
  cmake --build "${BUILD_DIR}" -j"$(nproc)"
fi

if [[ ${DO_BUILD} -eq 0 ]]; then
  echo "Next steps:"
  echo "  cmake -S \"${INFERENCE_DIR}\" -B \"${BUILD_DIR}\""
  echo "  cmake --build \"${BUILD_DIR}\" -j\$(nproc)"
fi
