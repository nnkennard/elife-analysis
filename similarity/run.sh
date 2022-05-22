
mkdir pdfs xmls
python prep_disapere_data.py
java -Xmx4G -jar /work/nnayak_umass_edu/grobid-0.7.1/grobid-core/build/libs/grobid-core-0.7.1-onejar.jar \
    -gH /work/nnayak_umass_edu/grobid-0.7.1/grobid-home \
    -dIn pdfs/ \
    -dOut xmls/ \
    -exe processFullText

