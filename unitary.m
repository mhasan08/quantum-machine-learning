
%
%   @author Munawar Hasan <munawar.hasan@nist.gov>
%

function [U] = unitary(n)
  X = rand(n) + 1i*rand(n);
  [U,~] = qr(X);
endfunction