# BLAST your sequences against ORGSP-1.0

wget https://rapdb.dna.affrc.go.jp/download/archive/irgsp1/IRGSP-1.0_protein_2021-11-11.fasta.gz
gunzip IRGSP-1.0_protein_2021-11-11.fasta.gz

# cat Oryza_sativa.IRGSP-1.0.pep.all.fa|sed 's/ .*//g' > os_pep.fa
cat IRGSP-1.0_protein_2021-11-11.fasta |sed 's/t/g/g'|sed 's/ .*//g' > os_pep.fa

## blastp
makeblastdb -in os_pep.fa -dbtype prot -out ospep.prot

for ptm in Kac Kcr Khib Kmal Ksucc Kla; do blastp -query ${ptm}_protein.fa -db ospep.prot -outfmt 6 -max_target_seqs 20 -out os_${ptm}.blastp -num_threads 128 -evalue 1e-5; done

for ptm in Kac Kcr Khib Kmal Ksucc Kla
do
    less os_${ptm}.blastp |awk '{if ($3>99) print $0}'|awk '{print $2}'|sed 's/-.*//g'|grep 'Os' > input/${ptm}_genes.txt
done