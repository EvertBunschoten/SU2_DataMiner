#!/bin/sh -l

echo "PINNTraining Docker Container"
usage="$(basename "$0") [-h] [-b branch_name]
where:
    -h  show this help text
    -b  branch name (if not given, existing PINNTraining directory must be mounted in /src/PINNTraining).

Compiled binaries can be found at /install/. Mount that directory for access.
Note: If you specify a working directory using the --workdir option for docker,
      append this directory to all paths above (e.g. use --workdir=/tmp if running in user mode)."

flags=""
branch=""
workdir=$PWD

export CCACHE_DIR=$workdir/ccache

if [ "$#" -ne 0 ]; then
  while [ "$(echo $1 | cut -c1)" = "-" ]
    do
        case "$1" in
            -b)
                    branch=$2
                    shift 2
                ;;
            *)
                    echo "$usage" >&2
                    exit 1
                ;;
    esac
    done
fi


if [ ! -z "$branch" ]; then
  name="SU2_DataMiner_$(echo $branch | sed 's/\//_/g')"
  echo "Branch provided. Cloning to $PWD/src/$name"
  if [ ! -d "src" ]; then
    mkdir "src"
  fi
  cd "src"
  git clone --recursive https://github.com/EvertBunschoten/SU2_DataMiner $name
  cd $name
  git config --add remote.origin.fetch '+refs/pull/*/merge:refs/remotes/origin/refs/pull/*/merge'
  git config --add remote.origin.fetch '+refs/heads/*:refs/remotes/origin/refs/heads/*'
  git fetch origin
  git checkout $branch
  git submodule update
  export SU2DATAMINER_HOME=$PWD
else
  if [ ! -d "src/SU2_DataMiner" ]; then
    echo "SU2_DataMiner source directory not found. Make sure to mount existing SU2_DataMiner at directory at /src/SU2_DataMiner. Otherwise use -b to provide a branch."
    exit 1
  fi
  cd src/SU2_DataMiner
  export SU2DATAMINER_HOME=$PWD
fi

export PYTHONPATH=$PYTHONPATH:$SU2DATAMINER_HOME
export PATH=$PATH:$SU2DATAMINER_HOME/bin/