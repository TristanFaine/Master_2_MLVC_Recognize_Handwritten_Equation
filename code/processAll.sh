#!/bin/bash


if [ $# -lt 2 ]
then
	echo "Apply all recognition step to all inkml files and produce the LG files"
	echo "Copyright (c) H. Mouchere, 2018"
	echo ""
	echo "Usage: processAll <input_inkml_dir> <output_lg_dir>"
	echo ""
	exit 1
fi

INDIR=$1
OUTDIR=$2

if ! [ -d $OUTDIR ] 
then
	mkdir $OUTDIR
	mkdir $OUTDIR/hyp
	mkdir $OUTDIR/seg
	mkdir $OUTDIR/symb
  mkdir $OUTDIR/result
fi

for file in $1/*.inkml
do
	echo "Recognize: $file"
	BNAME=`basename $file .inkml`
	OUT="$OUTDIR"
	python3 segmenter.py -i $file -o $OUT/hyp/$BNAME.lg
	ERR=$? # test de l'erreur au cas où...
	 python3 segmentSelect.py -o $OUT/seg/$BNAME.lg  $file $OUT/hyp/$BNAME.lg
	ERR=$ERR || $?
	 python3 symbolReco.py  -o $OUT/symb/$BNAME.lg $file $OUT/seg/$BNAME.lg
	ERR=$ERR || $?
	 python3 selectBestSeg.py -o $OUT/result/$BNAME.lg $OUT/symb/$BNAME.lg
	ERR=$ERR || $?
	
	if [ $ERR -ne 0 ]
	then 
		echo "erreur !" $ERR
		exit 2
	fi
done
echo "done."