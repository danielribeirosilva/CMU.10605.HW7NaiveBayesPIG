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
TestDocs = LOAD 'rcv1_test/small/RCV1.small_test_wids.txt' USING PigStorage() AS (id:int, labels:chararray, text:chararray);


--------------------------------------------------------------------------------
-- PART 1: TRAINING
--------------------------------------------------------------------------------


----------------------------------------
-- 1.1 COMPUTE TERM P(Y=y)
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
-- 1.2 COMPUTE TERM #(W=*,Y=y)
----------------------------------------


----------------------------------------
--split words and labels  
----------------------------------------

EventCountSplit = FOREACH EventsCount GENERATE STRSPLIT(wl,'\u002C') AS wl:(word,label), count as count;

----------------------------------------
-- compute  #(W=*,Y=y)
----------------------------------------

EventCountGroup = GROUP EventCountSplit BY wl.label;
LabelWordCount = FOREACH EventCountGroup GENERATE group as label, SUM(EventCountSplit.count) AS word_count;


----------------------------------------
-- 1.3 SMOOTHING TERM 
----------------------------------------

----------------------------------------
-- get vocabulary size for smoothing
----------------------------------------

EventCountGroupWord = GROUP EventCountSplit BY wl.word;
WordList = FOREACH EventCountGroupWord GENERATE group as word;
WordGroup = GROUP WordList ALL;
VocabSize = FOREACH WordGroup GENERATE COUNT(WordList) as vocabulary_size;

----------------------------------------
-- add smoothing to #(W=*,Y=y)
----------------------------------------

LabelWordCountAndVocabSize = CROSS LabelWordCount, VocabSize;
LogSmoothedLabelWordCount = FOREACH LabelWordCountAndVocabSize GENERATE label, LOG(word_count + vocabulary_size) AS log_smoothed_word_count;


--------------------------------------------------------------------------------
-- PART 2: TESTING
--------------------------------------------------------------------------------

----------------------------------------
-- 2.1 TOKENIZE DOCUMENTS
----------------------------------------

SplitAllTest = FOREACH TestDocs GENERATE id, TOKENIZE(labels) AS split_labels, TOKENIZE(text) AS split_text;
TestWordLabel = FOREACH SplitAllTest GENERATE FLATTEN(split_labels) AS label, FLATTEN(split_text) AS word;

