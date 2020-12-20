# __Data engineering GCP__
## __목차__  
[- Using BigQuery to do Analysis](#using-bigquery-to-do-analysis)  
[- Loading Taxi Data into Google Cloud SQL](#loading-taxi-data-into-google-cloud-sql)  
[- Loading data into BigQuery](#loading-data-into-bigquery)  
[- Working with JSON and Array data in BigQuery](#working-with-json-and-array-data-in-bigquery)  
[- Runnig Apache Spark jobs on Cloud Dataproc](#runnig-apache-spark-jobs-on-cloud-dataproc)  
[- Building and Executing a Pipeline Graph with Data Fusion](#building-and-executing-a-pipeline-graph-with-data-fusion)  
[- An Introduction to Cloud Composer](#an-introduction-to-cloud-composer)  
[- Streaming Data Processing: Publish Streaming Data into PubSub](#streaming-data-processing:-publish-streaming-data-into-pubSub)  
[- Streaming Data Processing: Streaming Data Pipelines](#streaming-data-processing:-streaming-data-pipelines)   
[- Streaming Data Processing: Streaming Analytics and Dashboards](#streaming-data-processing:-streaming-analytics-and-dashboards)  
[- Streaming Data Processing: Streaming Data Pipelines into Bigtable](#streaming-data-processing:-streaming-data-pipelines-into-bigtable)  
[- Optimizing your BigQuery Queries for Performance](#optimizing-your-bigquery-queries-for-performance)  
[- Partitioned Tables in Google BigQuery](#partitioned-tables-in-google-bigquery)  
[- Using the Natural Language API to classify unstructured text](#using-the-natural-Language-api-to-classify-unstructured-text)  
[- BigQuery in Jupyter Labs on AI Platform](#bigquery-in-jupyter-labs-on-ai-platform)  
[- Running AI models on Kubeflow](#running-ai-models-on-kubeflow)  
[- Predict Bike Trip Duration with a Regression Model in BQML](#predict-bike-trip-duration-with-a-regression-model-in-bqml)  
[- Movie Recommendations in BigQuery ML](#movie-recommendations-in-bigquery-ml)  
</br></br>

## __Using BigQuery to do Analysis__
---
크게 GCP 의 4가지 장점은,
  - Severless
  - Ease of use
  - Scale
  - Shareability

Bigquery는 자유롭게 interactive query을 수행하여, 데이터 작업을 진행할 수 있고, 더불어 여러 데이터 set을 합치고 자유롭게 분석 가능하다
</br></br>

## __Loading Taxi Data into Google Cloud SQL__
---
Cloud shell을 이용해 Cloud SQL instance을 실행 할 수 있다. _gcloud sql_, _mysql 명령어들을 이용해 작업이 가능하다.

요지는 mySQL 같은 것을 기존에 사용한 유저라면, 쉽게 gcp에서도 할 수 있다는 것
</br></br>

## __Loading data into BigQuery__
---
Bigquery는 다양한 source로 부터 데이터를 로딩 할 수 있다.
- 직접 만들기 (craate a new dataset and table)
- local 파일 업로드
- Cloud storage 로부터 로딩 (shell에서 bq load 명령어)
- 다른 table로 부터 생성 (CREATE TABLE, FROM 활용)
</br></br>


## __Working with JSON and Array data in BigQuery__
---
### __Array__
BigQuery natively supports arrays
- Array values must share a data type (Data in an array [ ] must all be the same type
)
- Arrays are called REPEATED fields in BigQuery

Creating your own arrays with ARRAY_AGG()
- finding the number of elements with ARRAY_LENGTH(<array>)
- deduplicating elements with ARRAY_AGG(DISTINCT <field>)
- ordering elements with ARRAY_AGG(<field> ORDER BY <field>)
- limiting ARRAY_AGG(<field> LIMIT 5)

Querying datasets that already have ARRAYs
- You need to UNNEST() arrays to bring the array elements back into rows
- UNNEST() always follows the table name in your FROM clause (think of it conceptually like a pre-joined table)

### __Struct__
A STRUCT is to consider it conceptually like a separate table that is already pre-joined into your main table. A STRUCT can have:
- one or many fields in it
- the same or different data types for each field
- it's own alias

Recap of STRUCTs:

- A SQL STRUCT is simply a container of other data fields which can be of different data types. The word struct means data structure. Recall the example from earlier:
  - STRUCT("Rudisha" as name, [23.4, 26.3, 26.4, 26.1] as splits) AS runner

- STRUCTs are given an alias (like runner above) and can conceptually be thought of as a table inside of your main table.

- STRUCTs (and ARRAYs) must be unpacked before you can operate over their elements. Wrap an UNNEST() around the name of the struct itself or the struct field that is an array in order to unpack and flatten it. 

Storing your large reporting tables as STRUCTs (pre-joined "tables") and ARRAYs (deep granularity) allows you to:
- gain significant performance advantages by avoiding 32 table JOINs
- get granular data from ARRAYs when you need it but not be punished if you dont (BigQuery stores each column individually on disk)
- have all the business context in one table as opposed to worrying about JOIN keys and which tables have the data you need
</br></br>

## __Runnig Apache Spark jobs on Cloud Dataproc__
---
기존 local 에서 작업하던 Spark jobs을 Cloud Dataproc에 migrate 할 수 있는지 가이드를 주는 예제이다.
우선, 3가지 단계로 진행된다.
  1. Dataproc 내에서 단순히 Spark jobs을 가져와서 실행한다. 이 경우, Hadoop과 spark가 포함된 이미지로 Dataproc cluster을 생성 후, HDFS에 데이터 복사 후, 작업 실행한다.
  1. Compute 영역과 Storage 영역을 분리해서 실행. 즉, HDFS가 아닌 Cloud storage에서 Spark jobs을 수행한다. 실제 이부분은 매우 간단히 동작하는데, 그냥 cloud storage에 있는 주소에 데이터를 지정하면 된다... 매우 쉽다!
  1. 마지막으로 Spark jobs을 Dataproc job에 deploy 하여 최적화 한다.
  우선 standalone Python file을 생성 후, Dataproc에서 이를 싱행하면 된다. (즉, notebook 기반의 방식이 아니라, standalone Python file로 생성 후, 돌리면 된다는 것을 말하는 듯 하다...)
</br></br>

## __간단 요약 정리__
---
GCP의 대부분 서비스의 경우 open source 기반의 관리 서비스
- Dataflow: Apache BEAM
- Dataproc: Hadoop, Spark
- Datafusion: CDAP
- DataComposer: Apach airflow
</br></br>

## __Building and Executing a Pipeline Graph with Data Fusion__
---
__Cloud Data Fusion__ is a service for __efficiently building ETL/ELT data pipelines__. Cloud Data Fusion __uses Cloud Dataproc cluster__ to perform all transforms in the pipeline.  
</br>
__Wrangler__ is __an interactive, visual tool__ that lets you see the effects of transformations on a small subset of your data before dispatching large, parallel-processing jobs on the entire dataset.

</br>
Cloud Data Fusion translates your visually built pipeline into an Apache Spark or MapReduce program that executes transformations on an ephemeral Cloud Dataproc cluster in parallel. This enables you to easily execute complex transformations over vast quantities of data in a scalable, reliable manner, without having to wrestle with infrastructure and technology.  
</br></br>

실습 진행시 특징:
  - Batch pipeline 과 realtime pipeline 모두 구성 가능
  - Wrangler을 이용해, block 들을 간단히 연결해 주는 것으로 쉽게 pipeline 구축 가능
  - 특히 여러 soruce로 부터 데이터들의 join/save 등이 매우 직관적이여서 편함
  - 다만 전반적으로 data fusion instance을 실행시키고, 동작시키는 속도가 빠르지는 않음
  - GUI로 실시간 Status 확인 가능
    - Provising-Starting-Running-Succeeded
    - in/Out/Error 갯수 확인 가능
</br></br>

## __An Introduction to Cloud Composer__
---
Cloud Composer is a hosted version of the popular open source workflow tool Apache Airflow

### __Airflow and core concepts__

Airflow is a platform to programmatically author, schedule and monitor workflows.
</br></br> 
Use Airflow to author workflows as directed acyclic graphs (DAGs) of tasks. The airflow scheduler executes your tasks on an array of workers while following the specified dependencies
</br></br> 
Core concepts
- DAG: A Directed Acyclic Graph is a collection of all the tasks you want to run, organized in a way that reflects their relationships and dependencies.
- Operator: The description of a single task, it is usually atomic. For example, the BashOperator is used to execute bash command.
- Task: A parameterised instance of an Operator; a node in the DAG.
- Task Instance: A specific run of a task; characterized as: a DAG, a Task, and a point in time. It has an indicative state: running, success, failed, skipped, ...

실습 진행시 특징:
  - Composer의 경우 instance 생성시 상당히 오래 걸림 (15~20분)
  - Workflow을 정의하기 위해서 python 코드 작성 필요: 해당 부분은 template 필요할듯
  - Airflow UI을 이용해 workflow을 visual하게 설정 가능
  - 동작원리를 정리해보면,  
    1. DAG 파일을 DAG folder (composer detail 확인) 복사
    1. Composer가 DAG을 airflow 에 추가하고, 자동적으로 스케쥴링
    1. DAG 반영 (3~5분정도)
    1. Composer의 경우 storage로 parsing

</br></br>
## __A Simple Dataflow Pipeline (Python)__
---
Apache Beam is an open source platform for executing data processing workflows.

실습 과정 특징
- apache beam 설치, 3가지 과정에 대해 추가되면서 진행  
  1. 단순하게 dataflow을 이용한 filtering
  1. MapReduce 적용을 통해 package split 후 combineKey()을 이용  
  GroupByKey() 와 비교해서 CombineKey() 추천-> 셔플링이 현저히 적어 속도 유리
  1. Side input 추가 예제 (local, dataflow 둘다 진행)  
  dataflow 진행시, job graph 확인 가능
- python에서 활용시, pipe을 이용해 비교적 쉽게 구현가능하나, apache beam에서 주로 사용되는 library는 어느정도 익숙해져야 될듯... 그리고 원하는 기능 구현시, template이 필요할 듯
</br></br>

## __Streaming Data Processing: Publish Streaming Data into PubSub__
---
Google Cloud Pub/Sub is a fully-managed real-time messaging service that allows you to __send and receive messages between independent applications.__ Use Cloud Pub/Sub to publish and subscribe to data from multiple sources, then use Google Cloud Dataflow to understand your data, all in real time.

실습 과정 특징  
- topic (publish)-subscription 간의 메시지 전달
- 상당히 사용이 간단하나, pulbish을 위한 코드는 작성이 필요
- Streaming sensor data을 publish하고, topic과 연결시킨 subscription instance을 만들고, 'pull' 명령어를 이용해, 실시간으로 메시지 받을 수 있음
- 한가지 문제점으로 보이는 부분은, publish와 subscription 간의 timing 문제가 발생가능해서, 못받거나 혹은 순서가 뒤바뀔 수 있는 가능성 존재  
  해결책: 강사가 언급했듯이, pub/sub 과 bigquery 등과 연결할 때, 중간에 dataflow 을 만들어, 시점/중복 여부 체크 필요
</br></br>

## __Streaming Data Processing: Streaming Data Pipelines__
---
앞서 언급한 PubSub을 이용해 streaming data을 얻을 때, 중간에 Dataflow pipeline을 두고 BigQuery로 rawdata을 저장하는 것을 구현

몇가지 중요 작업 포인트를 보면,
  - Pipeline 구성  
    주요 순서는 다음과 같다 (답이 있는 것은 아니다. 예제임)
      1. GetMessages from PubSub topic 
      1. Extract Data
      1. Time window: 데이터를 일정 window동안 accumulate 해서 전달하기 위해 설정
      1. 추가적인 Transform 적용
      1. 각 정보에 대해 row 생성
      1. BigQuery table로 데이터 전달 및 쓰기

  - SSH 터미널에서 Dataflow 코드(java) 실행시키면, 이를 콘솔 Dataflow에서 visual하게 확인 가능. 또한, 파이프의 노드를 클릭해보면 system lag 과 elemnets added 등과 같은 정보를 쉽게 파악 가능

  - 이 과정에서 자동적으로 auto-scaling 작업이 진행가능. 이는 Dataflow-JOB METRICS 항목에 Autoscaling section에서 확인 가능

  - 각종 Metrics 정보는 __Monitoring__ 메뉴에서 더 자세히 확인이 가능하며, 추가적으로 alert와 같은 기능도 포함 가능
    - Metric Explorer
    - Dashboards
    - Alerting

참고) 용어  
- _Data watermark lag_: The age (time since event timestamp) of the most recent item of data that has been fully processed by the pipeline.
- _System lag_: The current maximum duration that an item of data has been awaiting processing, in seconds.
</br></br>

## __Streaming Data Processing: Streaming Analytics and Dashboards__
---
Google Data Studio을 통해 데이터 visualization 하는 방법 소개
Data studio는 Google cloud 환경과 별도의 서비스로, 새로운 웹 브라우저에서 실행하며, connector을 이용해 연동을 해야 함

연동된 data에 대해 표 뿐만 아니라 다양한 차크를 그릴 수 있고, SQL query을 이용해 custom metric도 추출 가능

그리고 Google cloud 에서 (Bigquery을 예로 들면) query history 체크를 통해 Data studio에서 custom metric을 추출하기 위해 사용한 query문도 확인 가능

__개인적인 생각__: 보고서 작성이 목적이 아니라면,,, 굳이 사용할까???
Dashboard 보다는 훨씬 보고자료 만들기는 좋을듯~

</br></br>

## __Streaming Data Processing: Streaming Data Pipelines into Bigtable__
---
### __Apache HBase__ 란? (https://dydwnsekd.tistory.com/4)

Hadoop 플랫폼을 위한 구글의 BigTable을 본보기로 자바를 기반으로 만들어진 데이터 비관계형 분산 데이터 베이스이다.

NoSQL로 분류되며 스키마 지정 및 변경 없이 데이터를 저장할 수 있으며 하둡의 분산 파일 시스템인 HDFS에서 동작하기 때문에 가용성 및 1)확장성을 그대로 이용할 수 있다.

구글의 BigTable 논문에 설명된 2)Bloom필터 기능을 제공하며 자바를 기반으로 만들어져 자바 API, REST, Avro, Thrift를 통해 접근 가능

### __실습 내용__
이전과 비교해서 Bigquery 사용하는 것과 특별히 달라지는 것은 없다.
마찬가지로 Dataflow을 이용해, BigTable로 전달이 되는 구조인데, 한가지 특징은 위에 HBase shell prompt 을 이용해 query 작업을 한다는 것이다.

BigTable에 저장된 데이터를 보면, NoSQL이라서, 각 row는 column, timestamp, value combination으로 구분된다.
</br></br>

## __Optimizing your BigQuery Queries for Performance__
---
Query 실행시간과 비용을 줄이는데 있어 다음과 같은 종류의 테크닉이 있다.
  - Minimizing I/O
  - Caching results of previous queries
  - Performing efficient joins
  - Avoid over-whelming single workers
  - Using approximate aggregation functions

### __Minimizing I/O__
3개의 column의 합을 구하는 것과 2개의 column의 합을 구하는 경우 성능차이가 발생하는데, 주 이유는 데이터 읽는 양이 늘어나기 때문이다. 
__즉, 간단한 query에 overhead는 계산보다는 I/O에 의한 것이다.__
다음의 3가지를 고려해보자.
  1. Be purposeful in SELECT  
    SELECT * EXCEPT 나 필요한 column만 읽자
  1. Reduce data being read
    Column 데이터를 잘 보고, 기준이 되는 column을 잘 정한다. 즉, 어떤 기준으로 group by을 진행하냐에 따라, Repartition 작업이 추가되고, 이로 인한 속도 저하가 발생
  1. Reduce number of expensive computations
    예를 들어 두 테이블을 join 한 후에, 거리계산 등과 같은 것을 꽤 비싼 연산으로, 하나의 테이블에서 먼저 거리 계산을 마친 상태에서 join시 값을 읽어 오는 식으로 진행하면, shuffling등을 줄일 수 있게 되어 slot time의 감소를 가지고 온다
    
### __Cache results of previous queries__
BigQuery 서비스는 약 24시간안에 동일한 query에 대해서는 다시 연산을 수행하지 않고, cache로 저장된 결과를 반환한다. 

하지만, 기본적으로 string 비교를 기준으로 동일 여부를 검사하다 보니, 몇몇 상황에 대해 cache miss을 야기하게 된다. 예를 들어 white space 같은 경우나, non-deteministic behavior가 포함된 경우 (Current time stamp 혹은 Rand) 등이다.

다음의 경우 cache 사용을 통해 속도 향상을 이룰 수 있는데,
  1. Cache intermediate results
    종종 쓰는 query 결과에 대해 결과를 table 혹은 materalized view로 저장. 이럴 경우 중간 테이블을 주기적으로 해줘야 하는데, 이 부분을 잘 고려해서, 성능이 나은 쪽으로 정하면 될듯
  1. Accelerate queries with BI Engine
    BI Engine의 경우 데이터중 relevant한 부분을 자동적으로 메모리에 저장하는데, 이러한 부분을 잘 활용하면, BI Engine하에, BigQuery의 cache 성능을 활용가능하다. 이 때, 한가지 유의 사항은 메모리와 dataset이 같은 region에 있어야 한다.

### __Efficient joins__
두개의 테이블을 join 하는 것은 slot간 communication bandwidth 한계에 의해 자연스럽게 성능 저하를 일으키게 되는데, 결국 join의 횟수를 줄이는 것이 성능 향상을 하게 한다.

  1. Denormalization  
     간단하다, 테이블들을 다 따로 관리한다. 결국 이것은 storage <-> join 비용 간의 trade-off이다
  1. Avoid self-joins of large tables
     일반적으로 aggregation이나 window function을 활용해 self-join은 피하자. 두번 읽거나 하지말고, 한번 읽은뒤 recast 한다 던지...
  1. Reduce data being joined  
     사전에 grouping을 할 수 있으면 해서, join할 데이터 갯수를 줄이자
  1. Use a window function instead of a self-join  
     예를 들어, 자전거 rent 사업에서 자전거의 보관 시간을 알고 싶은데, 보관 시작 시점, 보관 완료 시점(rent 시작 시점)의 정보가 있을 때, self-join을 사용하지 말고, window function (TIMESTAMP_DIFF, OVER) 을 이용하면 쉽게 해결 가능
  1. Join with precomputed values  
     반복 계산을 해야되거나, 혹은 작은 table에서 계산이 가능한 부분은 미리 계산을 해놓으면 좋다. 위의 __Reduce data being joined__ 와 어느정도 겹치는 부분 존재

### __Avoid overwhelming a worker__
Ordering 같은 몇몇 operations은 보통 single worker에서 진행되는데, 너무 큰 사이즈의 데이터를 ordering 하면 memory가 감당되지 않아, “resources exceeded” 에러 메지시를 나타내기도 한다.

  1. Limiting large sorts  
    예를 들어, 전체 기간동안 sorting을 하지 않고, day 혹은 week 단위로 sorting을 한다던지...
  1. Data skew  
    하나의 key값이 다른 key값들 보다 월등히 많이 있는 상황에서 GROUP BY와 함께 ARRAY_AGG을 수행하게 되면 위와 같은 문제가 발생. 일단 이런경우 많은 worker에 대해 분산해서 group 데이터를 만들고, 그 결과를 aggregate 하는 방식을 적용한다.  
    ex) timezone 하나에 대해서만 group by을 하지 말고, timezone과 repo_name 두개에 대해 groupby을 수행하고, 각 timezone별로 repo결과를 aggregate 하는 식으로

### __Approximate aggregation functions__
Bigquery에는 큰 stream 데이터에 대해 COUNT(DISTINCT...) 대신에 APPROX_COUNT_DISTINCT 같은 것을 제공한다. 이릉 이용!
</br></br>

## __Partitioned Tables in Google BigQuery__
---
만약 매우 큰 사이즈의 dataset이 있는 경우, 이를 이용해서 query를 수행하면, 시간과 비용이 많이 드는데 patitioned dataset을 만들어 놓으면, 이러한 문제를 줄일 수 있다.

예를 들어, 어떤 특정 시점의 데이터만 추출해서 검색을 하고 싶을 때, WHERE (datetime)을 적용하면 되는데, 해당 시점의 데이터가 없더라도 query문을 실행시키면 전부 searching을 하게 된다. 

만약 datetime 기준으로 partitioned table을 구성해 놓으면 (PARTITION BY 활용), 해당 날짜의 데이터가 없을 때 실제 process는 0B 만 수행하게 되어 비용을 아낄수 있다. 즉, 없는 데이터는 그냥 검색 안한다.
</br></br>

## __Using the Natural Language API to classify unstructured text__
---
Cloud Natural Language API 을 이용해서 sentiment/syntatic 분석과 text의 catergory 분석등을 할 수 있다. Google Cloud Natural Language API을 enable 시키고, API Key 을 받은 뒤, API을 사용할 수 있다.

Interactive하게 API call 해서 바로 확인하는 방법은, 다음과 같다
  1. request.json 파일을 만듬 ("document", "type", "content" 포함)
  2. cloud shell에서 __curl__ 명령어를 이용해 API로 메세지 보냄
  3. 결과 확인 

Large dataset에서는 이러한 부분을 python script로 작성하여 수행가능하고, 그 결과를 BigQuery 등에 저장 가능하다. Python 기준으로, google.cloud 라이브러리에서 language_v1을 import 하여 API 사용 가능하다.

앞서 봤듯이, BigQuery로 raw 결과를 저장하면, 간단한 query로 우리가 원하는 형태로 분석 결과를 추출 가능하다.
</br></br>

## __BigQuery in Jupyter Labs on AI Platform__
---
AI Platform 서비스(?) 에서 Jupyter notebook을 사용해서 BigQuery을 활용할 수 있다.
실제, 내가 가장 유용하게 사용할 수 있는 서비스 인것 같다^^

BigQuery을 Jupyter notebook에서 사용하기 위해서는 google-cloud-bigquery을 먼저 설치한다.
Jupyter notebook에서는 __%%__ 을 이용하는 __Magic function__ 이 있는데, 바로 __%%__ 을 활용해서 BigQuery 결과를 dataframe에 저장할 수 있다
```
%%bigquery df
SELECT
*
FROM
<dataset>
```
즉, '아래 query을 실행하고, 그 결과를 df에 저장하라' 뜻이다.
</br></br>

## __Running AI models on Kubeflow__
---
__Kubenetes__  
Multiple nodes 에 거쳐 distributed ML을 가능케 하는 Platform으로써, 개발자는 마치 하나의 하드웨어에서 사용하는 것처럼 여러개의 cluster에 program을 deploy할 수 있다. Kubenetes을 활용해서 개발자는 자유롭게 resource을 추가 제거 할 수 있고, train과 모델 serve에 모두 활용 가능하다.

__Kubeflow__  
Kubeflow는 Kubenetes 위에서 ML workload을 실행시키기 위한 목적으로 만들어진 __open-source project__로써, 간단하고, portable/scalable 하다. 다양한 ML task을 수행가능하며, Jupyter notebook 실행도 가능하고, Custom Resource Definition (CRDs)을 cluster에 추가하여 Kubenetes API의 확장 활용도 가능하다.   
Kubeflow는 deployment 관리를 위해 __Kustomize__ 라고 불라는 tool을 사용한다. 
또한 __kfctl__ 이라고 Kubeflow의 CLI가 있는데, 이것이 cluster에서 Kubeflow을 설치할 수 있게 한다. 마찬가지로 download/설치 필요!  
</br></br>
Cluster을 만드는 과정이 조금 복잡한데, 다음의 순서로 진행된다고 보면 된다
  1. Application 디렉토리 생성
  1. 설치를 위한 환경 변수 세팅
  1. Configuration 적용 후 Cluster 설치  

Configuration 정보는 YAML 파일에 저장한다.

위 설치 과정이 진행되면, GCP에 두곳에서 object들이 생성되는데,
  1. Deployment Manger: kubeflow-storage, kubeflow
  1. Kubernetes Engine: Cluster가 보이고, Kubeflow 컴퍼넌트 갯수와 서비스 갯수를 각각 __Workloads__ 와 __Service & Ingress__ 세션에서 확인 가능하다.   

Cluster가 정상적으로 설치되면, __kubectl__ 을 이용해서 communcation이 가능하다.

설치과 정상적으로 끝났으면, 본격적으로 Training 을 진행해 보자  
Training을 진행하기 위한 과정을 간략히 정리하면, 다음과 같다.
  1. Starge Bucket 생성
  1. Container 빌드 (Docker image build)  
     Container 빌드 후, 간단하게 local에서 실행하여 정상적으로 동작하는지 확인
  1. Google Container Registry (GCR)에 업로드

GCR에 정상적으로 이미지 업로딩이 진행되면, cluster에서 training이 가능하다.  
Training은 다음과 같이 진행된다.
  1. Application 폴더로 이동
  1. YAML 파일에 학습에 사용될 파라미터 세팅  
    _kustomize edit add_ 명령어를 이용해 추가도 가능
  1. Bucket내 Read/Write 권한 관련해서 설정 필요
  1. 최종적으로 지정한 cloud storage에 checkpoint들과 serving을 위한 export가 생성  
    중간에 ```kubectl logs -f my-train-1-chief-0``` 와 같은 명령어로 CLI에서 진행 확인 가능

학습이 정상적으로 완료가 되면, Serving 작업을 진행 할 수 있다. (이건 선택의 문제겠지?)  
마찬가지로 과정을 간단히 설명하면,
  1. Kubeflow manifests 파일 내에 serving implementation 정보가 포함되는데, 여기에 GCS bucket의 모델 위치를 지정한다.
  1. 그러면 request을 처리하는 server가 spin up 된다. 한가지 특징은, 학습때와 달리 custom container가 필요하지 않다. 
  1. Web UI로 deploy도 가능
</br></br>

## __Predict Bike Trip Duration with a Regression Model in BQML__  
---
BigQuery Machine Learning (BQML) 은 데이터 분석/학습/평가/예측등을 다 지원하는 서비스.
BigQuery을 이용해 다양한 feature을 추출 가능하며, _Explore with Data Studio_ 을 활용해 chart을 이용해 visual하게 feature 분석이 가능하다

모델 역시, BigQuery 안에서 만들 수 있는데, 예를 들면
```sql
CREATE OR REPLACE MODEL
  bike_model.model
OPTIONS
  (input_label_cols=['duration'],
    model_type='linear_reg') AS
SELECT
  duration,
  start_station_name,
  CAST(EXTRACT(dayofweek
    FROM
      start_date) AS STRING) AS dayofweek,
  CAST(EXTRACT(hour
    FROM
      start_date) AS STRING) AS hourofday
FROM
  `bigquery-public-data`.london_bicycles.cycle_hire
```
와 같이 _CREATE OR REPLACE MODEL_, _OPTIONS_ 을 이용해 간단한 모델 적용이 가능하다.
_OPTIONS_ 에는 label 열과 어떤 모델을 택할 것인지를 정한다.

또 재밌는 부분은 Model을 생성할 때, _TRANSFORM_ 구문을 이용해 feature transform을 지정하면, 우리가 prediction 할 때, 자동적으로 feature의 transform까지 진행해서 수행한다.
```sql
CREATE OR REPLACE MODEL
  bike_model.model_bucketized TRANSFORM(* EXCEPT(start_date),
  IF
    (EXTRACT(dayofweek
      FROM
        start_date) BETWEEN 2 AND 6,
      'weekday',
      'weekend') AS dayofweek,
    ML.BUCKETIZE(EXTRACT(HOUR
      FROM
        start_date),
      [5, 10, 17]) AS hourofday )
OPTIONS
  (input_label_cols=['duration'],
    model_type='linear_reg') AS
...
...
```
TRANSFORM이 정의된 모델을 가지고 다음과 같이 prediction을 수행한다.
```sql
SELECT
  *
FROM
  ML.PREDICT(MODEL bike_model.model_bucketized,
    (
    SELECT
      start_station_name,
      start_date
    FROM
      `bigquery-public-data`.london_bicycles.cycle_hire
    LIMIT
      100) )
```
참고로 위의 예시는 100개 batch prediction을 수행하는 경우이다.

학습된 모델의 파라미터(모델 weight)도 쉽게 BigQuery을 이용해 확인이 가능하다
```sql
SELECT * FROM ML.WEIGHTS(MODEL bike_model.model_bucketized)
```
</br></br>

## __Movie Recommendations in BigQuery ML__  
---
BigQuery Data을 터미널에서 로딩하는 방법은 다음과 같다.
```bash
bq --location=EU mk --dataset movies

bq load --source_format=CSV \
--location=EU \
--autodetect movies.movielens_ratings \
gs://dataeng-movielens/ratings.csv
```

이번 실습 예제가 이전과 다른 부분은 __Collaborative filtering__ 에 대해서 다룬다는 점이다.
__Collaborative filtering__ 이란, 비슷한 user 또는 아이템을 찾는 방법 중 하나로, 비슷한 유저들의 rating 을 기반으로 rating이 없는 혹은 새로운 사용자에 대한 rating을 계산하는 방법이다.

이러한 방법이 필요한 이유는, 우리가 다루는 데이터에 일부는 공란으로 제공되는 경우가 있는데, 이러한 Collaborative filtering을 이용해서 해당 부분을 채울수도 있다.

마지막 두 실습의 경우 결국 BQML에서 데이터 분석/학습/Prediction 등을 어떻게 하는지를 보여주는 예제였다. 우선적으로 SQL 구문에 익숙하다면, 쉽게 데이터를 추출하고 학습까지 진행가능하다. 























