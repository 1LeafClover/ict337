# 🚀 ICT337 - Big Data Computing in the Cloud (AWS EMR)

**PySpark solutions for SUSS assignments**  
*Tutor-Marked Assignments (TMA) & End-Course Assessment (ECA)*  

[![AWS EMR](https://img.shields.io/badge/AWS-EMR-orange)](https://aws.amazon.com/emr/)
[![PySpark](https://img.shields.io/badge/PySpark-3.3.1-red)](https://spark.apache.org/)

## 📂 Solutions Overview

### TMA (All Questions)
- **Flight Data Analysis**: 200+ metrics on `flights_data.csv` using DataFrames
- **Market Basket**: Association rules on `grocery_data.csv` using RDDs
- **Core Concepts**: RDD vs DataFrames comparison
- **EMR Setup**: Cluster configuration docs

### ECA (All Questions) 
- **Movie Recommendation**: ALS implementation on `mov_*.dat` files
- **Vehicle MPG**: Manufacturer analysis on TSV/CSV data  
- **K-Means**: Custom clustering implementation
- **Spark Fundamentals**: Hadoop vs Spark deep dive

## ⚡ EMR Cluster Specs
```yaml
Managed Scaling: Enabled
Min Capacity: 2 units 
Max Capacity: 8 units

Node Types:
- 1 x Core (m4.large)
- 1 x Task (m4.large)

Max Limits:
- 8 core nodes
- 8 on-demand instances

Applications:
- Spark 3.3.1
- Hadoop 3.3.3
