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
TestDocs = LOAD 'rcv1_test/small/RCV1.small_test_wids.txt' USING PigStorage() AS (id:int, true_labels:chararray, text:chararray);


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

-- STORE AllTrainingData INTO 'training_data/AllTrainingData' using PigStorage('\t');
-- LOAD 'training_data/AllTrainingData/' USING PigStorage('\t') AS (label:chararray, word_label_count:long, log_smoothed_word_count:double, label_loglikelihood:double);




----------------------------------------
-- 1.5 AUXILIARY DATA
----------------------------------------
----------------------------------------
-- Compute Label List
----------------------------------------
-- LabelsGroup = GROUP EventCount BY label;
-- LabelsList = FOREACH LabelsGroup GENERATE group AS label;
----------------------------------------
-- Add Label Info
----------------------------------------
-- LabelWithDataListJoin = JOIN LabelsList BY label, LabelTrainingData BY label;
-- LabelWithDataList = FOREACH LabelWithDataListJoin GENERATE LabelTrainingData::label AS label




--------------------------------------------------------------------------------
-- PART 2: TESTING
--------------------------------------------------------------------------------




----------------------------------------
-- 2.1 TOKENIZE DOCUMENTS
----------------------------------------

SplitAllTest = FOREACH TestDocs GENERATE id, true_labels, TOKENIZE(text) AS split_text;
TestWordLabel = FOREACH SplitAllTest GENERATE id, true_labels, FLATTEN(split_text) AS word;


----------------------------------------
-- 2.2 GENERATE TEST DATA
----------------------------------------

----------------------------------------
-- get word count per doc on test set
----------------------------------------

TestWordDocLabelGroup = GROUP TestWordLabel BY (id, true_labels, word);
TestWordDocCount = FOREACH TestWordDocLabelGroup GENERATE FLATTEN(group.id) AS id, group.true_labels AS true_labels,FLATTEN(group.word) AS word, COUNT(TestWordLabel) AS word_count;

----------------------------------------
-- link each (id,word) with each label
----------------------------------------

TestWordDocAllLabelsCross = CROSS TestWordDocCount, LabelTrainingData;
TestWordDocAllLabels = FOREACH TestWordDocAllLabelsCross GENERATE id, true_labels, word, word_count, label, log_smoothed_word_count, label_loglikelihood; 


----------------------------------------
-- 2.3 GET WORD INFO
----------------------------------------

TestWordInfoJoin = JOIN TestWordDocAllLabels BY (word,label) LEFT OUTER, EventCount BY (word,label);
TestWordInfo = FOREACH TestWordInfoJoin GENERATE id, TestWordDocAllLabels::TestWordDocCount::word AS word, TestWordDocAllLabels::LabelTrainingData::label AS label, true_labels, word_count AS test_word_count, log_smoothed_word_count, label_loglikelihood, word_label_count AS train_word_count;

----------------------------------------
-- 2.4 ADD SMOOTHING
----------------------------------------

SmoothedData = FOREACH TestWordInfo GENERATE id, word, label, true_labels, test_word_count, log_smoothed_word_count, label_loglikelihood, (train_word_count is null ? 1L : ((long)train_word_count+1L)) AS smoothed_train_word_count;

----------------------------------------
-- 2.5 COMPUTE SCORES
----------------------------------------

----------------------------------------
-- word scores
----------------------------------------

TestScore = FOREACH SmoothedData GENERATE id, word, label, true_labels, (test_word_count*(LOG(smoothed_train_word_count)-log_smoothed_word_count)) AS word_label_loglikelihood, label_loglikelihood;

----------------------------------------
-- group by label
----------------------------------------
TestScoreGroup = GROUP TestScore BY (id,label,true_labels);
ScoresByDocLabel = FOREACH TestScoreGroup GENERATE group.id AS id, group.label AS label, group.true_labels AS true_labels, SUM(TestScore.word_label_loglikelihood) AS words_label_loglikelihood, MAX(TestScore.label_loglikelihood) AS label_loglikelihood;







