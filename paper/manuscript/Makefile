# Build targets (run inside paper/manuscript/).
#   make        -> main.pdf and ../supplementary/supplementary.pdf (two passes for cross-references)
#   make anon   -> main_anonymous.pdf and ../supplementary/supplementary_anonymous.pdf
#   make bib    -> refresh refs.bib from the verified project bibliography
#   make md     -> refresh ../manuscript.md and ../supplementary.md (review mirrors)
#   make lint   -> style lint of both mirrors (needs the project style/ folder)
#   make clean  -> remove LaTeX build files
LINT ?= ../../style/lint.py
PROFILE ?= ../../style/style_thresholds.json
SUPP = ../supplementary

all:
	latexmk -pdf -interaction=nonstopmode main.tex
	cd $(SUPP) && latexmk -pdf -interaction=nonstopmode supplementary.tex
	latexmk -g -pdf -interaction=nonstopmode main.tex
	cd $(SUPP) && latexmk -g -pdf -interaction=nonstopmode supplementary.tex

anon: all
	sed 's/^\\anonymousfalse/\\anonymoustrue/' main.tex > main_anonymous.tex
	latexmk -pdf -interaction=nonstopmode main_anonymous.tex && latexmk -c main_anonymous.tex && rm -f main_anonymous.tex main_anonymous.bbl
	cd $(SUPP) && sed 's/^\\anonymousfalse/\\anonymoustrue/' supplementary.tex > supplementary_anonymous.tex && latexmk -pdf -interaction=nonstopmode supplementary_anonymous.tex && latexmk -c supplementary_anonymous.tex && rm -f supplementary_anonymous.tex supplementary_anonymous.bbl

bib:
	python3 ../tools/extract_bib.py

md: all
	python3 ../tools/tex2md.py all

lint: md
	python3 $(LINT) ../manuscript.md --profile $(PROFILE) --out ../lint_latest_main.md
	python3 $(LINT) ../supplementary.md --profile $(PROFILE) --out ../lint_latest_supplementary.md

clean:
	latexmk -c main.tex
	cd $(SUPP) && latexmk -c supplementary.tex

.PHONY: all anon bib md lint clean
