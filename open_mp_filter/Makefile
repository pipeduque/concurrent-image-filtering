open_mp_filter_program: open_mp_filter.c
	/opt/homebrew/Cellar/llvm/17.0.6/bin/clang -Xpreprocessor -I/opt/homebrew/Cellar/libomp/17.0.6/include -L/opt/homebrew/Cellar/libomp/17.0.6/lib -fopenmp open_mp_filter.c -o open_mp_filter_program -lomp

clean:
	rm -f open_mp_filter_program
