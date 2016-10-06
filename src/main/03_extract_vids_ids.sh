cd ../dataset2

files=`ls | grep "[0-9]b\?.txt"`

for fname in $files; do
    sed 's/.*_\([0-9]*\)-.*/\1/g' $fname | sort -u > ${fname%.*}_vids.txt
done
