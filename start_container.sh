#!/bin/bash

nvidia-docker run --rm -it -v `pwd`:/src -p 8888:8888 -p 4567:4567 mkolod/udacity_carnd_term1_gpu /bin/bash
