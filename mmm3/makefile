CXX = icpc
CXXFLAGS = -qopenmp -mkl
KNCFLAGS = $(CXXFLAGS) -mmic
CPUFLAGS = $(CXXFLAGS) -xhost
OPTFLAGS = -qopt-report=5 -qopt-report-file=$@.optrpt
TIME = 'date +%T-%D'

all: mic-app xeon-app

mic-app: main.cpp
	$(CXX) $(KNCFLAGS) $(OPTFLAGS) -o "$@" "$<"

xeon-app: main.cpp
	$(CXX) $(CPUFLAGS) $(OPTFLAGS) -o "$@" "$<"

run-mic:
	micnativeloadex ./mic-app
	#scp mic-app mic0:~/
	#MIC-AnhTuTran
	#ssh mic0 ./mic-app > out.`date +%Y.%m.%d-%H.%M.%S`

run-xeon:
	./xeon-app
	#export SINK_LD_LIBRARY_PATH=/opt/share/intel/compilers_and_libraries/linux/lib/mic/:/opt/share/intel/compilers_and_libraries_2017.4.196/linux/mkl/lib/mic/

clean:
	rm -f *app *.optrpt
	#ssh mic0 rm -f MIC-AnhTuTran/mic-app
