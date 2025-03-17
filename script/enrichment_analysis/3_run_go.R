library(GO.db)
library(dplyr)
library(clusterProfiler)
library(ggplot2)
library(AnnotationHub)
library(data.table)
library(gridExtra)
library(patchwork)
library(purrr)
# Define gene files and respective directories
gene_files <- c("Kac_genes.txt", "Kcr_genes.txt", "Khib_genes.txt", "Kmal_genes.txt", "Ksucc_genes.txt", "Kla_genes.txt")
output_dir <- "output"
figures_dir <- "figures"
cache_dir <- "cache"

# Create output directories if they don't exist
dir.create(output_dir, showWarnings = FALSE)
dir.create(figures_dir, showWarnings = FALSE)
dir.create(cache_dir, showWarnings = FALSE)

hub <- AnnotationHub()
query(hub, c("Oryza sativa","orgdb"))
rice <- hub[["AH114586"]]

all_plots <- list()
all_enrich <- list()
# Loop through each gene list file
for (gene_file in gene_files) {
  
  # Read gene list and set directories for current gene set
  genes <- read.table(paste0("input/", gene_file), header=FALSE)
  genes <- genes[[1]]
  
  # Set subfolder for each gene list
  gene_name <- gsub("_genes.txt", "", gene_file)
  gene_output_dir <- paste0(output_dir, "/", gene_name)
  gene_figures_dir <- paste0(figures_dir, "/", gene_name)
  dir.create(gene_output_dir, showWarnings = FALSE)
  dir.create(gene_figures_dir, showWarnings = FALSE)
  
  # Step 1: GO Term ID to GO Term Name Mapping Tables
  go_table <- as.data.frame(GOTERM)
  godb <- unique(go_table[,c(1,3,4)])
  godb_BP <- godb[godb$Ontology == "BP", 1:2]
  godb_MF <- godb[godb$Ontology == "MF", 1:2]
  godb_CC <- godb[godb$Ontology == "CC", 1:2]
  
  # Save results to cache
  write.table(godb_BP, paste0(cache_dir, "/godb_BP_", gene_name, ".txt"), sep = "\t", quote = FALSE, row.names = FALSE)
  write.table(godb_MF, paste0(cache_dir, "/godb_MF_", gene_name, ".txt"), sep = "\t", quote = FALSE, row.names = FALSE)
  write.table(godb_CC, paste0(cache_dir, "/godb_CC_", gene_name, ".txt"), sep = "\t", quote = FALSE, row.names = FALSE)

  # Step 2: Gene ID to GO Term ID Mapping
  goid2gene_all <- read.table(paste0(cache_dir, "/rice_combined_go_list.tsv"), header=T, sep="\t", quote="", stringsAsFactors = F)
  goid2gene_BP <- goid2gene_all %>% filter(go_id %in% godb_BP$go_id)
  goid2gene_MF <- goid2gene_all %>% filter(go_id %in% godb_MF$go_id)
  goid2gene_CC <- goid2gene_all %>% filter(go_id %in% godb_CC$go_id)
  
  # Save results to cache
  write.table(goid2gene_BP, paste0(cache_dir, "/goid2gene_BP_", gene_name, ".txt"), sep="\t", row.names=FALSE, quote=FALSE)
  write.table(goid2gene_MF, paste0(cache_dir, "/goid2gene_MF_", gene_name, ".txt"), sep="\t", row.names=FALSE, quote=FALSE)
  write.table(goid2gene_CC, paste0(cache_dir, "/goid2gene_CC_", gene_name, ".txt"), sep="\t", row.names=FALSE, quote=FALSE)

  # Step 3: GO enrichment analysis with clusterProfiler
  # find rice orgDb
  genes <- genes
  # ID conversion table
  IDtable <- read.csv("input/riceIDtable.csv")
  genes_eid <- IDtable[match(genes, IDtable$rapdb), "entrezgene"]
  # remove NAs
  genes_eid <- as.character(genes_eid[!is.na(genes_eid)])
  # Background genes (use all genes from the dataset or provide your own list)
  bkgd <- read.table(paste0("input/", 'bkgd.txt'), header=FALSE)
  bkgd <- bkgd[[1]]
  
  ## Step 4: GO enrichment analysis using annotations from AnnotationHub package
  # Run GO enrichment analysis
  ## run enrichGO function
  go2 <- enrichGO(gene = genes_eid, # a vector of gene id
                OrgDb = rice, # OrgDb object
                ont = "all", # One of "MF", "BP", and "CC" subontologies
                pvalueCutoff = 0.9, # p-value cutoff (default)
                pAdjustMethod = "BH", # multiple testing correction method to calculate adjusted p-value (default)
                qvalueCutoff = 0.9, # q-value cutoff (default). q value: local FDR corrected p-value.
                minGSSize = 10, # minimal size of genes annotated for testing (default)
                maxGSSize = 500  # maximal size of genes annotated for testing (default)
                )
  # Save GO results
  go2_df <- as.data.frame(go2)
  #数据处理#
  go2_df$term <- paste(go2_df$ID, go2_df$Description, sep = ': ') #将ID与Description合并成新的一列
  go2_df$term <- factor(go2_df$term, levels = go2_df$term,ordered = T) #转成因子，防止重新排列
  go_enrich <- go2_df %>%
    group_by(ONTOLOGY) %>%
    arrange(desc(Count), .by_group = TRUE) %>%
    slice_head(n = 10)
  go_enrich <- go_enrich %>%
    mutate_at(vars(pvalue, p.adjust, qvalue), ~ ifelse(. > 0.05, 0.01, .))
  write.table(go_enrich, paste0(gene_output_dir, "/go_df_", gene_name, ".txt"), sep="\t", row.names=FALSE, quote=FALSE)
  
  #纵向柱状图——-根据pvalue值绘制#
  #纵向柱状图-根据ONTOLOGY类型绘制#
  p1 <- ggplot(go_enrich,
    aes(x=term,y=Count, fill=ONTOLOGY)) + #x、y轴定义；根据ONTOLOGY填充颜色
    geom_bar(stat="identity", width=0.8) +  #柱状图宽度
    scale_fill_manual(values = c("#6666FF", "#33CC33", "#FF6666") ) +  #柱状图填充颜色
    xlab("") +  #x轴标签
    ylab("") +  #y轴标签
    coord_flip() +  #让柱状图变为纵向
    theme_bw()
  #根据ONTOLOGY分类信息添加分组框#
  # Generate GO enrichment plot
  p1 <- p1+facet_grid(ONTOLOGY~., scale = 'free_y', space = 'free_y')
  all_plots[[gene_name]] <- p1
  all_enrich[[gene_name]] <- go_enrich
}
final_combined_plot <- wrap_plots(all_plots, ncol = 3)

ggsave(final_combined_plot, filename = paste0(figures_dir, "/go_dotplots.pdf"), height = 50, width = 100, units = "cm")


all_enrich <- imap(all_enrich, ~ .x %>% mutate(Group = .y))
all_enrich <- lapply(all_enrich, function(df) {
    df %>% mutate(term = as.character(term))
})
combined_go_enrich <- bind_rows(all_enrich)
write.table(combined_go_enrich, paste0(output_dir, "/combined_all_enrich.csv"), sep=",", row.names=FALSE, quote=FALSE)
p <- ggplot(combined_go_enrich,
       aes(y=term,x=Group))+
    geom_point(aes(size=Count,color=pvalue))+   #Count表示气泡图中点的大小，并按pvalue值进行着色
    scale_color_gradient(low = "#EDE4FF",high ="#7C00FE")+
    labs(color=expression(PValue,size="Count"), 
         x="Acylation type",y="",title="")+
    theme_bw()
p <- p+facet_grid(ONTOLOGY~., scale = 'free_y', space = 'free_y') +
    theme(
        strip.background = element_rect(fill = "#D7BBF5", color = "black"),  # 修改分面框背景色和边框颜色
        strip.text = element_text(color = "black", size = 12, face = "bold")  # 修改分面框文本样式
    )
ggsave(p, filename = paste0(figures_dir, "/Final_go_dotplots.pdf"), height = 30, width = 30, units = "cm")
ggsave(p, filename = paste0(figures_dir, "/Final_go_dotplots.png"), height = 30, width = 30, units = "cm")
