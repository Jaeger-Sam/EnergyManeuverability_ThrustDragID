% LP_15smooth.fcn is a 15 point digital low pass Spencer filter.
% See Equations 2.13 & 2.17 in Flight Vehicle System Identification by
% Jategaonkar.
%
% y = LP_15smooth(u)
%
% INPUTS:
%   u: noisy signal
%
% OUTPUTS:
%   y: filtered signal
%
% Sam Jaeger
% 11/4/2025

function y = LP_15smooth(u)
    y = u;
    for nn=3:(length(u)-2)
        if nn<8 || nn>(length(u)-8)
            y(nn) = (7*u(nn-2) + 24*u(nn-1) + 34*u(nn) + 24*u(nn+1) + 7*u(nn+2))/96;
        else
            y(nn) = (-3*u(nn-7) - 6*u(nn-6) - 5*u(nn-5) + 3*u(nn-4) + 21*u(nn-3) ...
                + 46*u(nn-2) + 67*u(nn-1) + 74*u(nn) + 67*u(nn+1) + 46*u(nn+2) ...
                + 21*u(nn+3) + 3*u(nn+4) - 5*u(nn+5) - 6*u(nn+6) - 3*u(nn+7))/320;
        end
    end
end