# 📋 Data Preprocessing Methods Documentation

---

### 🚀 Overview:
This document provides a detailed summary of the various data preprocessing methods we've ventured into during our research.

---

### 🧪 Methods Explored:

- **Chipper Cleaned Data Set**: 
    - Description: Utilized the Chipper cleaned data set.
    
- **Raw BAM Files**: 
    - Description: Worked with raw BAM files without any prior cleaning.
    
- **Min-Max Reads (0 to 1)**: 
    - Description: Applied Min-Max normalization for reads between 0 and 1.
    
- **Excluding Reads**: 
    - Description: Chose to omit the reads when passing data to our transformer.
    
- **Min-Max Reads (0 to 1000)**: 
    - Description: Adapted Min-Max normalization for reads to have a range between 0 and 1000, to amplify the potential impact.
    
- **Individual Normalization**: 
    - Description: Executed normalization for each data point individually from 0 to 1000, opposed to a global approach, with the aspiration to highlight the shape of the ATAC-seq peak.

---

### 🔍 Observations:

🚫 **No Significant Improvement**: Regardless of the method, none showcased a marked enhancement over the others.

📉 **Reads Impact**: The absence of reads in the model yielded a similar outcome, indicating a negligible influence of reads on our predictions.

---

### 🔜 Next Steps:

💡 Given the observations, there's an evident need to brainstorm methods that could potentially augment the impact of reads on our predictions. This might necessitate a deeper examination of the dataset or pondering over architectural modifications for heightened sensitivity to reads.

---

