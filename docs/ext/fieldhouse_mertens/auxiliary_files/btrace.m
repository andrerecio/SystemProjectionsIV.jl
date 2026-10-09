function X = btrace(Y,H)
NR = size(Y,1)/H;
NC = size(Y,2)/H;

X = NaN(NR,NC);
    for r = 1:NR
        for c = 1:NC
            X(r,c) = trace(Y(1+(r-1)*H:r*H,1+(c-1)*H:c*H));
        end
    end
end