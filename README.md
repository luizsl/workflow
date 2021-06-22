# README

This file uses the markdown syntax, it can be converted to an elegant pdf or HTML with some application that manage such format. After the compilation # , ## and so on are converted to a chain of hierarchical titles and subtitles. But it's readable even without any compilation.

## Directory architecture

```
.
├──── data
│     ├──── <galaxy name>
│     │     ├──── Muse
│     │     ├──── Hubble
│     │     ╰──── Spitzer
│     ├──── models
│     ╰──── misc_data 
├──── data_products
├──── exploration
├──── miscellaneous
├──── plots
├──── results
╰──── src
```

- data\
It contains 'raw' data, mainly muse data processed by the timer team. It also can contain Hubble or Spitzer data. The contents of this folder is too large for Dropbox and also private from the public in some cases so it won't be on Git.

- data_products\
Data that are result of the analysis and can also be used in subsequent steps as input (Also private for now).

- exploration\
Some code draft. After tested, the code is moved to the src directory to be used in the main analysis.

- miscellaneous\
Store files that are not suited in another folder. This folder should preferally be empty.

- plots\
To save plots produced by code residing in the src directory.

- results\
Documents for publication.

- src\
Directory for code which are tested, debugged and whose results are trustworthy.

These folders can contain their own readme files if necessary.
