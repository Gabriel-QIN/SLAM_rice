library(GO.db)
library(dplyr)
library(clusterProfiler)
library(ggplot2)
library(AnnotationHub)
library(data.table)
library(gridExtra)
library(patchwork)

# Define gene files and respective directories
gene_files <- c("Kac_genes.txt", "Kcr_genes.txt", "Khib_genes.txt", "Kmal_genes.txt", "Ksucc_genes.txt", "Kla_genes.txt")
output_dir <- "output"
figures_dir <- "figures"
cache_dir <- "cache"

# Create output directories if they don't exist
dir.create(output_dir, showWarnings = FALSE)
dir.create(figures_dir, showWarnings = FALSE)
dir.create(cache_dir, showWarnings = FALSE)

all_plots <- list()
all_enrich <- list()
# Loop through each gene list file
for (gene_file in gene_files) {
  #   Step 4: KEGG enrichment analysis
  #   Read the rice gene annotation table (large file)
  anno <- as.data.frame(fread("cache/IRGSP-1.0_representative_annotation_2021-11-11.tsv", quote=""))
  # Read gene list and set directories for current gene set
  genes <- read.table(paste0("input/", gene_file), header=FALSE)
  genes <- genes[[1]]
  
  # Set subfolder for each gene list
  gene_name <- gsub("_genes.txt", "", gene_file)
  gene_output_dir <- paste0(output_dir, "/", gene_name)
  gene_figures_dir <- paste0(figures_dir, "/", gene_name)
    # Background genes (use all genes from the dataset or provide your own list)
  bkgd <- read.table(paste0("input/", 'bkgd.txt'), header=FALSE)
  bkgd <- bkgd[[1]]
  
  #   Convert the gene IDs to match with KEGG database
  genes_transID <- anno[match(genes, anno$Locus_ID), "Transcript_ID"]
  bkgd_transID <- anno[match(bkgd, anno$Locus_ID), "Transcript_ID"]
  
  genes_transID <- as.character(genes_transID[!is.na(genes_transID)])
  bkgd_transID <- as.character(bkgd_transID[!is.na(bkgd_transID)])
  
  genes_transID <- gsub("-.*", "", genes_transID)
  bkgd_transID <- gsub("-.*", "", bkgd_transID)

  # Run KEGG enrichment analysis
  kegg <- enrichKEGG(gene = genes_transID,
                     universe = bkgd_transID,
                     organism = "dosa", # KEGG organism code for Oryza sativa
                     keyType = "kegg",  # Key type for KEGG
                     pvalueCutoff = 0.9,
                     pAdjustMethod = "BH",
                     qvalueCutoff = 0.9,
                     minGSSize = 10,
                     maxGSSize = 500)
  kegg@result$Description = gsub(" - Oryza.*", "", kegg@result$Description)
  kegg@result$Description = gsub(" - plant", "", kegg@result$Description)
  kegg@result$Description = gsub(" - other", "", kegg@result$Description)
  # Save KEGG results
  kegg_df <- as.data.frame(kegg)
  kegg_enrich <- kegg_df %>%
    arrange(desc(Count), .by_group = TRUE) %>%
    slice_head(n = 20)
  kegg_enrich <- kegg_enrich %>%
    mutate_at(vars(pvalue, p.adjust, qvalue), ~ ifelse(. > 0.05, 0.01, .))
  write.table(kegg_enrich, paste0(gene_output_dir, "/kegg_df_", gene_name, ".txt"), sep="\t", row.names=FALSE, quote=FALSE)
  
  # Generate KEGG enrichment plot
  p2 <- dotplot(kegg, showCategory=10,
                font.size = 10)
  ggsave(p2, filename = paste0(gene_figures_dir, "/kegg_dotplot_", gene_name, ".png"), height = 12, width = 22, units = "cm")
  all_plots[[gene_name]] <- p2
  all_enrich[[gene_name]] <- kegg_enrich
}
final_combined_plot <- wrap_plots(all_plots, ncol = 3)
ggsave(final_combined_plot, filename = paste0(figures_dir, "/all_gene_sets_kegg_dotplots.png"), height = 20, width = 60, units = "cm")

all_enrich <- imap(all_enrich, ~ .x %>% mutate(Group = .y))
combined_kegg_enrich <- bind_rows(all_enrich)
write.table(combined_kegg_enrich, paste0(output_dir, "/combined_all_kegg_enrich.csv"), sep=",", row.names=FALSE, quote=FALSE)
p <- ggplot(combined_kegg_enrich,
       aes(y=Description,x=Group))+
    geom_point(aes(size=Count,color=pvalue))+   #Count表示气泡图中点的大小，并按pvalue值进行着色
    scale_color_gradient(low = "#EDE4FF",high ="#7C00FE")+
    labs(color=expression(PValue,size="Count"), 
         x="Acylation type",y="",title="")+
    theme_bw()
ggsave(p, filename = paste0(figures_dir, "/Final_kegg_dotplots.pdf"), height = 20, width = 20, units = "cm")
ggsave(p, filename = paste0(figures_dir, "/Final_kegg_dotplots.png"), height = 20, width = 20, units = "cm")
# End of script
