--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- Naive Bayes Classification - Apache PIG
-- Code for specific label ECAT
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
-- get doc count per label: #(Y=y)  105,667
----------------------------------------
LabelDocCountGroup = GROUP ClassEventsCount BY label;
LabelDocCount = FOREACH LabelDocCountGroup GENERATE group as label, SUM(ClassEventsCount.count) as label_doc_count;
ECATDocCount = FILTER LabelDocCount BY (label == 'lab=ECAT');

----------------------------------------
-- get total doc size:  #(Y=*)  2,323,118
----------------------------------------

DocCountGroup = GROUP LabelDocCount ALL;
DocCount = FOREACH DocCountGroup GENERATE SUM(LabelDocCount.label_doc_count) as total_doc_count;

----------------------------------------
-- divide to get prior:  P(Y=y) = #(Y=y) / #(Y=*)  -> log = -3.0902782556383306
----------------------------------------

BothDocCounts = CROSS ECATDocCount,DocCount; 
ECATPriorLog = FOREACH BothDocCounts GENERATE label, LOG((double)label_doc_count / (double)total_doc_count) AS prior_log;


----------------------------------------
-- 1.2 COMPUTE TERM #(W=*,Y=ECAT)
----------------------------------------


----------------------------------------
--split words and labels  
----------------------------------------

EventCountSplit = FOREACH EventsCount GENERATE STRSPLIT(wl,'\u002C') AS wl:(word,label), count as count;
EventCount = FOREACH EventCountSplit GENERATE FLATTEN(wl), count AS word_label_count;

----------------------------------------
-- get #(W=w,Y=ECAT) in right format
----------------------------------------

EventCountECAT = FILTER EventCount BY (label == 'lab=ECAT');
EventCountECATWordSplit = FOREACH EventCountECAT GENERATE STRSPLIT(word,'=') AS wordsplit:(prefix,word), label, word_label_count;
EventCountECATAdjusted = FOREACH EventCountECATWordSplit GENERATE FLATTEN(wordsplit), label, word_label_count;
EventCountECATAdjusted2 = FOREACH EventCountECATAdjusted GENERATE word, label, word_label_count;


----------------------------------------
-- compute  #(W=*,Y=ECAT)
----------------------------------------

EventCountSplitECAT = FILTER EventCountSplit BY (wl.label == 'lab=ECAT');
EventCountGroup = GROUP EventCountSplitECAT BY wl.label;
ECATWordCount = FOREACH EventCountGroup GENERATE group as label, SUM(EventCountSplitECAT.count) AS word_count;


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
-- add smoothing to #(W=*,Y=ECAT) and LOG
----------------------------------------

ECATWordCountAndVocabSize = CROSS ECATWordCount, VocabSize;
ECATLogSmoothedWordCount = FOREACH ECATWordCountAndVocabSize GENERATE label, LOG((double)word_count + (double)vocabulary_size) AS log_smoothed_word_count;


----------------------------------------
-- 1.4 PUT ALL DATA TOGETHER
----------------------------------------

----------------------------------------
-- Join #(W=*,Y=ECAT) and logP(Y=ECAT)
----------------------------------------

ECATTrainingDataJoin = JOIN ECATLogSmoothedWordCount BY label, ECATPriorLog BY label;
ECATTrainingData = FOREACH ECATTrainingDataJoin GENERATE ECATLogSmoothedWordCount::ECATWordCount::label AS label, log_smoothed_word_count, prior_log;

--(lab=ECAT,17.19608453158187,-3.0902782556383306)



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
-- link each (id,word) with ECAT data
----------------------------------------

TestWordDocECATCross = CROSS TestWordDocCount, ECATTrainingData;
TestWordDocECAT = FOREACH TestWordDocECATCross GENERATE id, true_labels, word, word_count, label, log_smoothed_word_count, prior_log; 


----------------------------------------
-- 2.3 GET TRAINING WORD INFO
----------------------------------------

TestWordInfoJoin = JOIN TestWordDocECAT BY word LEFT OUTER, EventCountECATAdjusted2 BY word;
TestWordInfo = FOREACH TestWordInfoJoin GENERATE id, TestWordDocECAT::TestWordDocCount::word AS word, TestWordDocECAT::ECATTrainingData::label AS label, true_labels, word_count AS test_word_count, log_smoothed_word_count, prior_log, word_label_count AS train_word_count;

----------------------------------------
-- 2.4 ADD SMOOTHING
----------------------------------------

SmoothedData = FOREACH TestWordInfo GENERATE id, word, label, true_labels, test_word_count, log_smoothed_word_count, prior_log, (train_word_count is null ? 1L : ((long)train_word_count+1L)) AS smoothed_train_word_count;

----------------------------------------
-- 2.5 COMPUTE SCORES
----------------------------------------

----------------------------------------
-- word scores
----------------------------------------

TestScore = FOREACH SmoothedData GENERATE id, word, label, true_labels, (test_word_count*(LOG(smoothed_train_word_count)-log_smoothed_word_count)) AS word_label_loglikelihood, prior_log;

----------------------------------------
-- group by document
----------------------------------------
TestScoreGroup = GROUP TestScore BY (id,label,true_labels);
ScoresByDocLabel = FOREACH TestScoreGroup GENERATE group.id AS id, group.label AS label, group.true_labels AS true_labels, SUM(TestScore.word_label_loglikelihood) AS words_label_loglikelihood, MAX(TestScore.prior_log) AS prior_log;

----------------------------------------
-- add prior to result
----------------------------------------

FinalScores = FOREACH ScoresByDocLabel GENERATE id, label, true_labels, (words_label_loglikelihood + prior_log) AS likelihood;

----------------------------------------
-- sort by descending loglikelihood
----------------------------------------

FinalScoresSorted = ORDER FinalScores BY likelihood DESC;




