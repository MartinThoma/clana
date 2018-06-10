upload:
	make clean
	python3 setup.py sdist bdist_wheel && twine upload dist/*

clean:
	rm -rf clana.egg-info dist tests/reports tests/__pycache__ lidtk.errors.log lidtk.info.log clana/cm_analysis.html
