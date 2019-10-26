upload:
	make clean
	python3 setup.py sdist bdist_wheel && twine upload dist/*

clean:
	rm -rf clana.egg-info dist tests/reports tests/__pycache__ clana.errors.log clana.info.log clana/cm_analysis.html dist __pycache__ clana/__pycache__ build docs/build

mutmut-results:
	mutmut junitxml --suspicious-policy=ignore --untested-policy=ignore > mutmut-results.xml
	junit2html mutmut-results.xml mutmut-results.html
