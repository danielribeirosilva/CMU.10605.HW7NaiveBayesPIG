--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- Naive Bayes Classification - Apache PIG
-- Author: Daniel Ribeiro Silva
-- E-mail: drsilva@cs.cmu.edu
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------



----------------------------------------
-- LOAD DATA
----------------------------------------

ClassEventsCount = LOAD 'rcv1_train_class_events/part-0*' USING PigStorage() AS (label:chararray, count:long);
EventsCount = LOAD 'rcv1_train_events/part-0*' USING PigStorage() AS (wl:chararray, count:long);



--------------------------------------------------------------------------------
-- PART 1: TRAINING
--------------------------------------------------------------------------------


----------------------------------------
-- COMPUTE TERM P(Y=y)
----------------------------------------

----------------------------------------
-- get doc count per label: #(Y=y)
----------------------------------------

LabelDocCountGroup = GROUP ClassEventsCount BY label;
LabelDocCount = FOREACH LabelDocCountGroup GENERATE group as label, SUM(ClassEventsCount.count) as label_doc_count;

----------------------------------------
-- get total doc size:  #(Y=*)
----------------------------------------

DocCountGroup = GROUP LabelDocCount ALL;
DocCount = FOREACH DocCountGroup GENERATE SUM(LabelDocCount.label_doc_count) as total_doc_count;

----------------------------------------
-- divide to get probability:  P(Y=y) = #(Y=y) / #(Y=*)
----------------------------------------

BothDocCounts = CROSS LabelDocCount,DocCount; 
LabelLogLikelihood = FOREACH BothDocCounts GENERATE label, LOG((double)label_doc_count / (double)total_doc_count) AS loglikelihood;


----------------------------------------
-- COMPUTE TERM #(W=w,Y=y)
----------------------------------------


----------------------------------------
--split words and labels  
----------------------------------------

EventCountSplit = FOREACH EventsCount GENERATE STRSPLIT(wl,'\u002C') AS wl:(word,label), count as count;

----------------------------------------
-- get vocabulary size for smoothing
----------------------------------------




