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
LabelLogLikelihood = FOREACH BothDocCounts GENERATE label, LOG((double)label_doc_count / (double)total_doc_count) AS label_loglikelihood;


----------------------------------------
-- 1.2 COMPUTE TERM #(W=*,Y=y)
----------------------------------------

----------------------------------------
--split words and labels  
----------------------------------------

EventCountSplit = FOREACH EventsCount GENERATE STRSPLIT(wl,'\u002C') AS wl:(word,label), count as count;
EventCount = FOREACH EventCountSplit GENERATE FLATTEN(wl), count AS word_label_count;

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
-- add smoothing to #(W=*,Y=y) and LOG
----------------------------------------

LabelWordCountAndVocabSize = CROSS LabelWordCount, VocabSize;
LogSmoothedLabelWordCount = FOREACH LabelWordCountAndVocabSize GENERATE label, LOG((double)word_count + (double)vocabulary_size) AS log_smoothed_word_count;


----------------------------------------
-- 1.4 PUT ALL DATA TOGETHER
----------------------------------------

----------------------------------------
-- Join #(W=*,Y=y) and logP(Y=y)
----------------------------------------

LabelTrainingDataJoin = JOIN LogSmoothedLabelWordCount BY label, LabelLogLikelihood BY label;
LabelTrainingData = FOREACH LabelTrainingDataJoin GENERATE LogSmoothedLabelWordCount::LabelWordCount::label AS label, log_smoothed_word_count, label_loglikelihood;

----------------------------------------
-- Join result with #(W=w,Y=y)
----------------------------------------

AllTrainingDataJoin = JOIN LabelTrainingData BY label, EventCount BY label;
AllTrainingData = FOREACH AllTrainingDataJoin GENERATE word, LabelTrainingData::label AS label, word_label_count, log_smoothed_word_count, label_loglikelihood;





--------------------------------------------------------------------------------
-- PART 2: TESTING
--------------------------------------------------------------------------------




----------------------------------------
-- 2.1 TOKENIZE DOCUMENTS
----------------------------------------

SplitAllTest = FOREACH TestDocs GENERATE id, TOKENIZE(labels) AS split_labels, TOKENIZE(text) AS split_text;
TestWordLabel = FOREACH SplitAllTest GENERATE id, FLATTEN(split_labels) AS label, FLATTEN(split_text) AS word;


----------------------------------------
-- 2.2 COMPUTE LIKELIHOOD
----------------------------------------

----------------------------------------
-- get word count on test set
----------------------------------------

TestWordLabelGroup = GROUP TestWordLabel BY (id, label, word);
TestWordLabelCount = FOREACH TestWordLabelGroup GENERATE FLATTEN(group.id) AS id,FLATTEN(group.label) AS label,FLATTEN(group.word) AS word, COUNT(TestWordLabel) AS word_count;

----------------------------------------
-- merge training count with test count
----------------------------------------


