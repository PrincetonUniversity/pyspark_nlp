

## Getting Started
This example can be run as a Jupyter notebook or as a standalone application
through `pyspark_example.py` to demonstrate some of the power of Apache Spark.


To run this notebook, install the requirements listed in `requirements.txt` using Minconda3/Anaconda3 and
`conda create --name spark --file requirements.txt`. Then `source activate spark` to enable it.

This does not install Spark to run in standalone mode.
For that, you'll need to download and [install the
appropriate version of Spark for your OS](https://spark.apache.org/downloads.html).

## Running the example in standalone mode

This will produce a very efficient run (not as efficient as Apache's native
Scala, but what can you do?) of the same program. It is also much closer to what
you might do on an HPC cluster.

```{bash}
source activate spark
spark-submit pyspark_sample.py
```
The program should produce a great deal of output, and then produce a folder
with CSV output named `bigrams` and one with a full JSON output named `ngrams`.

