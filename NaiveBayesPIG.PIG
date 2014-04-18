--FOLDERS
--rcv1_test  
--rcv1_train_class_events  
--rcv1_train_events


--ssh drsilva@ghc23.ghc.andrew.cmu.edu

--export PATH=$PATH:/usr/local/hadoop/bin:/usr/local/pig-0.12.0/bin
--export JAVA_HOME=/usr/lib/jvm/jre-sun
--export CLASSPATH=`ls -1 /usr/local/hadoop/*.jar|perl -ne 'do{chop;print $sep,$_;$sep=":";}'`

pig -x local

ClassEventsCount = LOAD 'rcv1_train_class_events/part-0*' USING PigStorage() AS (label, count:long);
EventsCount = LOAD 'rcv1_train_events/part-0*' USING PigStorage() AS (label, count:long);
DUMP A;