mkdir temp
mkdir logs
mkdir data
cd data
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip 
unzip annotations_trainval2017.zip
rm annotations/captions* annotations/person*
rm annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip
unzip image_info_test2017.zip
mv annotations/image_info_test-dev2017.json annotations/instances_test-dev2017.json
rm image_info_test2017.zip
cd ..
mkdir data/preprocessed
mkdir data/detections
