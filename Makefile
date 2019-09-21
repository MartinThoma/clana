upload:
	make clean
	python3 setup.py sdist bdist_wheel && twine upload dist/*

clean:
	rm -rf clana.egg-info dist tests/reports tests/__pycache__ clana.errors.log clana.info.log clana/cm_analysis.html dist __pycache__ clana/__pycache__ build docs/build
