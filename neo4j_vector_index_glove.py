# Databricks notebook source
# MAGIC %md
# MAGIC # Vector Index Neo4j

# COMMAND ----------

# MAGIC %md
# MAGIC The following notebook experiments with the Neo4j Vector Index ([Neo4j’s Vector Search: Unlocking Deeper Insights for AI-Powered Applications](https://neo4j.com/blog/vector-search-deeper-insights/)). The experiments makese use of the [GloVe Embeddings](https://nlp.stanford.edu/projects/glove/) to perform vector search. 

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import types as T
from neo4j import GraphDatabase

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read vectors from Azure

# COMMAND ----------

# MAGIC %md
# MAGIC The vectors are obtained from [GloVe](https://nlp.stanford.edu/projects/glove/). For this the GloVe Vectors trained on Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 300d vector) were used. 

# COMMAND ----------

# MAGIC %md
# MAGIC The files are uploaded to a Storage Account in Azure. First retrieve the necessary Azure credentials. 

# COMMAND ----------

storage_account_name =  dbutils.secrets.get(scope="kv_db", key="saName")
storage_account_access_key =  dbutils.secrets.get(scope="kv_db", key="saKeyAccess")
spark.conf.set('fs.azure.account.key.' + storage_account_name + '.blob.core.windows.net', storage_account_access_key)

# COMMAND ----------

# MAGIC %md
# MAGIC Read the data

# COMMAND ----------

blob_container = "glove"
filePath = "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/glove.6B.300d.txt"
glove_df = spark.read.text(filePath)

# COMMAND ----------

glove_df.count()

# COMMAND ----------

display(glove_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocess Data

# COMMAND ----------

# MAGIC %md
# MAGIC To load it to Neo4j some preprocessing steps need to be taken. 

# COMMAND ----------

glove_df = (
    glove_df
    .withColumn('word', F.split(F.col('value'), ' ').getItem(0))
    .withColumn('vector', F.split(F.col('value'), ' '))
    .withColumn('vector', F.expr("slice(vector, 2, SIZE(vector))"))
    .withColumn('vector', F.transform(F.col("vector"), lambda x: x.cast("float")))
    .withColumn('vector_size', F.size(F.col('vector')))
    .withColumn('id', F.monotonically_increasing_id())
    .select('id', 'word', 'vector', 'vector_size')
)

# COMMAND ----------

glove_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Check whether all vector sizes are of length 300

# COMMAND ----------

assert glove_df.filter(F.col('vector_size') != 300).count() == 0

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load to Neo4j

# COMMAND ----------

# MAGIC %md
# MAGIC For loading to Neo4j the [Neo4j Spark Connector](https://neo4j.com/docs/spark/current/) will be used. 

# COMMAND ----------

# MAGIC %md
# MAGIC First set the credentials for Neo4j. Those are stored in Azure Key Vault.

# COMMAND ----------

database = "neo4j"
username = dbutils.secrets.get(scope="kv_db", key="neo4jAuraDSUsername")
password = dbutils.secrets.get(scope="kv_db", key="neo4jAuraDSPassword")
uri = dbutils.secrets.get(scope="kv_db", key="neo4jAuraDSuri")

# COMMAND ----------

# MAGIC %md
# MAGIC Write the Spark DataFrame as nodes to Neo4j. This can take a while depending on hardware configurations and the size of the dataset.

# COMMAND ----------

(
    glove_df
    .write
    .format("org.neo4j.spark.DataSource")
    .mode("Overwrite")
    .option("url", uri)
    .option("authentication.type", "basic")
    .option("authentication.basic.username", username)
    .option("authentication.basic.password", password)
    .option("database", database)
    .option("labels", ":Words")
    .option("node.keys", "id")
    .save()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Connect to Neo4j with Python Driver

# COMMAND ----------

# MAGIC %md
# MAGIC To run queries the [Neo4j Python Driver API](https://neo4j.com/docs/api/python-driver/current/api.html#api-documentation) is used.

# COMMAND ----------

# MAGIC %md
# MAGIC Set up driver connection and create functions with simple queries.

# COMMAND ----------

class App:
    def __init__(self, uri, user, password, database=None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password), database=database)
        self.database = database

    def close(self):
        self.driver.close()

    def query(self, query):
        return self.driver.execute_query(query)
        
    def count_nodes_in_db(self):
        query = "MATCH (n) RETURN COUNT(n)"
        result = self.query(query)
        (key, value) = result.records[0].items()[0]
        return value

    def remove_nodes_relationships(self):
        query = "MATCH (n) DETACH DELETE n"
        result = self.query(query)

    def print_records(self, result):
        for record in result.records:
            print(record.values())

# COMMAND ----------

# MAGIC %md
# MAGIC Setup a connection

# COMMAND ----------

app = App(uri, username, password, database)

# COMMAND ----------

app.count_nodes_in_db()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Indexes

# COMMAND ----------

# MAGIC %md
# MAGIC Before setting Vector Indices it is smart that for fast retrieval node indexes are set.

# COMMAND ----------

query = """
    CREATE INDEX node_range_index_id FOR (w:Words) ON (w.id)
"""

# COMMAND ----------

app.query(query)

# COMMAND ----------

query = """
    CREATE TEXT INDEX node_text_index_word FOR (w:Words) ON (w.word)
"""

# COMMAND ----------

app.query(query)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Vector Property

# COMMAND ----------

# MAGIC %md
# MAGIC Before we can create a Vector Index the property must be set as a vector property. This is done in batches. 

# COMMAND ----------

batch_size = 1000
nr_batches = int(app.count_nodes_in_db() / batch_size)
print(f'Running {nr_batches} batches with size {batch_size}')

# COMMAND ----------

for batch in range(nr_batches):
    query = f"""
        MATCH(n:Words)
        WHERE n.id >= {(batch*batch_size)+1} AND n.id <= {(batch+1)*batch_size}
        CALL db.create.setNodeVectorProperty(n, "vector", n.vector)
        RETURN count(n) AS propertySetCount
    """
    app.query(query)
    if batch % 100 == 0:
        print(f"Finished: {batch}/{nr_batches} batches ({round(batch/nr_batches*100,2)}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Vector Index

# COMMAND ----------

# MAGIC %md
# MAGIC Now the Vector Index can be set. 

# COMMAND ----------

query = """
    CREATE VECTOR INDEX `word-embeddings`
    FOR (w:Words) ON (w.vector)
    OPTIONS {indexConfig: {
        `vector.dimensions`: 300,
        `vector.similarity_function`: 'cosine'
    } }
"""

# COMMAND ----------

app.query(query)

# COMMAND ----------

query = """
    SHOW INDEX
"""

# COMMAND ----------

result = app.query(query)

# COMMAND ----------

app.print_records(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Search
# MAGIC

# COMMAND ----------

word = "database"

# COMMAND ----------

query = f"""
    MATCH (w:Words {{word: "{word}"}})
    WITH w
    CALL db.index.vector.queryNodes('word-embeddings', 5, w.vector)
    YIELD node AS similarWords, score
    RETURN similarWords.id, similarWords.word, score
"""
print(query)

# COMMAND ----------

result = app.query(query)

# COMMAND ----------

app.print_records(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare ANN with NN

# COMMAND ----------

query = f"""
    MATCH (n:Words {{word: "{word}"}})
    CALL {{ 
        WITH n
        MATCH (m:Words)
        RETURN m, gds.similarity.cosine(n.vector, m.vector) AS score ORDER BY score DESC LIMIT 5
    }}
    RETURN m.id, m.word, score 
"""
print(query)

# COMMAND ----------

result = app.query(query)

# COMMAND ----------

app.print_records(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare recall of ANN and NN

# COMMAND ----------

# MAGIC %md
# MAGIC Now compare the results of ANN and NN and measure the recall of both nearest neighbors.

# COMMAND ----------

query = f"""
    MATCH (w:Words)
    WITH w
    LIMIT 10
    CALL {{
        WITH w
        MATCH (n:Words)
        WITH n, gds.similarity.cosine(w.vector, n.vector) AS score ORDER BY score DESC LIMIT 10
        RETURN COLLECT(n.id) as nn
    }}
    CALL db.index.vector.queryNodes('word-embeddings', 10, w.vector)
    YIELD node AS ann
    WITH w, nn, COLLECT(ann.id) as ann
    RETURN w.id, gds.similarity.overlap(nn, ann) AS recall, nn, ann
"""
# print(query)

# COMMAND ----------

result = app.query(query)

# COMMAND ----------

app.print_records(result)
