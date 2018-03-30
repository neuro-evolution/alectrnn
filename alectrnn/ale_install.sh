#!/usr/bin/env bash

cd "$(dirname "$0")"
BASE_DIR=$(pwd)
INSTALL_DIR="$BASE_DIR/alelib"
# BIN_DIR="$INSTALL_DIR/bin"
# INCLUDE_DIR="$INSTALL_DIR/include"
# LIB_DIR="$INSTALL_DIR/lib"
# if [ ! -d "$BIN_DIR" ] || [ ! -d "$INSTALL_DIR" ] || [ ! -d "$LIB_DIR" ]; then

# Location of ALE fork used with ALECTRNN
ALE_PACKAGE_URL="https://github.com/nathaniel-rodriguez/arcade-learning-environment/archive/master.zip"
PACKAGE_FILE_NAME="master.zip"

SDL_FLAG="OFF"
LOCAL_LIB="OFF"
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -s|--with-sdl)
    SDL_FLAG="ON"
    ;;
    -l|--lib-path)
    LIBPATH="$2"
    LOCAL_LIB="ON"
    shift
    ;;
    *) break
  esac
  shift
done

if [ "$LOCAL_LIB" == "ON" ]; then
  echo "Using local ALE..."
  echo "copying from $LIBPATH"
  cp -r "$LIBPATH" "arcade-learning-environment-master"
elif [ "$LOCAL_LIB" == "OFF" ]; then
  # Download and extract
  if [ ! -x "$PACKAGE_FILE_NAME" ]; then
    wget -O $PACKAGE_FILE_NAME $ALE_PACKAGE_URL
  fi
  echo "extracting files . . . "
  unzip "$BASE_DIR/master.zip"
  rm -f "$BASE_DIR/master.zip"
fi

# Install ALE
if [ ! -d "$INSTALL_DIR" ]; then
  mkdir "$INSTALL_DIR"
fi

cd "arcade-learning-environment-master"
if [ ! -d "build" ]; then
  mkdir "build"
fi
cd "build"

if [ "$SDL_FLAG" == "OFF" ]; then
  echo "Installing without SDL..."
  CC=gcc CXX=g++ cmake -DUSE_SDL=OFF -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" ..
elif [ "$SDL_FLAG" == "ON" ]; then
  echo "Installing with SDL..."
  CC=gcc CXX=g++ cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" ..
fi

make
make install
cd "../.."
rm -rf "arcade-learning-environment-master"

# fi