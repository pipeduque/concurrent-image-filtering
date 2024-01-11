c_filter_program: c_filter.c
	/opt/homebrew/Cellar/llvm/17.0.6/bin/clang -Xpreprocessor -I/opt/homebrew/Cellar/libomp/17.0.6/include -L/opt/homebrew/Cellar/libomp/17.0.6/lib -fopenmp c_filter.c -o c_filter_program -lomp

clean:
	rm -f c_filter_program
