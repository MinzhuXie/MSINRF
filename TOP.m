 matPredict = load([".\data\interaction2.txt"]);
 cd_adjmat = load([".\smallrawdata1\Association matrix.txt"]);
circRNA_list=importdata([".\smallrawdata1\cicrRna.xlsx"]);
disease_list=importdata([".\smallrawdata1\Diease-name.xlsx"]);

[newNCP_rank,NCP_rank_known] =Rank_cicrRNAs( matPredict, cd_adjmat, circRNA_list, disease_list);
Write_file(newNCP_rank);
