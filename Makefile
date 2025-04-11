JULIA:=julia

default: help

setup:
	${JULIA} -e 'import Pkg; Pkg.add(["JuliaFormatter"])'

format:
	${JULIA} -e 'using JuliaFormatter; format(".")'

test:
	${JULIA} --project -e 'using Pkg; Pkg.resolve(); Pkg.test()'

docs:
	${JULIA} --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
	${JULIA} --project=docs -t auto docs/make.jl

vitepress:
	npm --prefix docs i
	npm --prefix docs run docs:dev

all: setup format test docs vitepress

help:
	@echo "The following make commands are available:"
	@echo " - make setup: install the dependencies for make command"
	@echo " - make format: format codes with JuliaFormatter"
	@echo " - make test: run the tests"
	@echo " - make docs: instantiate and build the documentation"
	@echo " - make vitepress: start Vitepress site of documentation"
	@echo " - make all: run every commands in the above order"

.PHONY: default setup format test docs vitepress all help