#! /bin/sh

root_dir=$1;
resolution=$2;
save_dir=$3;

sigmas_path=sigmas_$resolution.npy;
samples_path=samples_$resolution.npy;

for file in $(find $root_dir -name $sigmas_path); do
	echo "Processing $file";

	IFS='_'; #setting comma as delimiter
	read -a file_parts <<< $file; #reading str as an array as tokens separated by IFS
	IFS='';
	save_path="$save_dir/${file_parts[2]}/train/${file_parts[3]}_${file_parts[4]}_$sigmas_path";

	echo "Saving to $save_path";
	cp $file $save_path;
done

for file in $(find $root_dir -name $samples_path); do
	echo "Processing $file";

	IFS='_'; #setting comma as delimiter
	read -a file_parts <<< $file; #reading str as an array as tokens separated by IFS
	IFS='';
	save_path="$save_dir/${file_parts[2]}/train/${file_parts[3]}_${file_parts[4]}_$samples_path";

	echo "Saving to $save_path";
	cp $file $save_path;
done
