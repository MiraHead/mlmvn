#/bin/sh

NN_DIR=src/network/

make_doc:
	doxygen Doxyfile
	echo "Doxygen uses doxypy - see tag INPUT_FILTER in Doxyfile!!!."
	echo "You should install it from pip repos. in case you don't have it"
	
see_doc:
	firefox doc/html/index.html
