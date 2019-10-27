upload:
	make clean
	python3 setup.py sdist bdist_wheel && twine upload dist/*

clean:
	rm -rf clana.egg-info dist tests/reports tests/__pycache__ clana.errors.log clana.info.log clana/cm_analysis.html dist __pycache__ clana/__pycache__ build docs/build

mutmut-results:
	mutmut junitxml --suspicious-policy=ignore --untested-policy=ignore > mutmut-results.xml
	junit2html mutmut-results.xml mutmut-results.html

bandit:
	# Not a security application: B311 and B303 should be save
	# Python3 only: B322 is save
	bandit -r clana -s B311,B303,B322
