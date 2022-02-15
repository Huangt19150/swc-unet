function [precision, recall, f1] = compute_prf(pred, mask)
    
mutual = double(pred).* double(mask);
precision = (nnz(mutual)) / (nnz(pred) + 1e-8);
recall = (nnz(mutual))  / (nnz(mask) + 1e-8);
f1 = 2 * precision * recall / (precision + recall + 1e-8);

end