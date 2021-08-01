#!/bin/bash

sphinx-apidoc -f -o source/ ../filterpicker
make html
