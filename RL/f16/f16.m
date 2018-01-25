      function [XD,Y]=f16(T,X,U);
      
%     6-DOF model of F-16, see Stevens & Lewis page 125
%                     Copyright Jan. 1992  Brian L Stevens

      global XCG;
      persistent INIT S B CBAR RM XCGR HE C1 C2 C3 C4 C5 C6 C7 C8 C9 RTOD G XA;               
      if (isempty(INIT))
          S=300;
          B=30;
          CBAR=11.32;
          RM=1.57e-3;
          XCGR=0.35;
          HE=160.;
          C1=-.770;C2= .02755;C3=1.055E-4;C4= 1.642E-6;
          C5=.9604;C6=1.759E-2;C7=1.792E-5;C8=-.7336; C9=1.587E-5;
          RTOD=180./pi;
          G=32.17;
          XA=15.0;
          INIT=1;
      end;
      
      XD=zeros(13,1);
      
      VT=X(1);
      ALPHA=X(2)*RTOD;
      BETA =X(3)*RTOD;
      PHI  = X(4);
      THETA= X(5);
      PSI  = X(6);
      P    = X(7);
      Q    = X(8);
      R    = X(9);
      NORTH=X(10);
      EAST =X(11);
      ALT = X(12);
      if (ALT<0.) ALT=0.; end;
      POW  = X(13);

      THTL=U(1);
      AIL=U(2);
      EL=U(3);
      RDR=U(4);

%
      PHID  = PHI*RTOD;
      THETAD= THETA*RTOD;
      PD    = P*RTOD;
      QD    = Q*RTOD;
      RD    = R*RTOD;
%
% AIR DATA COMPUTER AND ENGINE MODEL
%
      [MACH,QBAR] = adc(VT,ALT);
      CPOW   =  TGEAR(THTL);
      XD(13) =  PDOT(POW,CPOW);
      T      =  THRUST(POW,ALT,MACH);
%
%  LOOK - UP TABLES AND COMPONENT BUILDUP
%
      CXT   = CX (ALPHA,EL);
      CYT   = CY (BETA,AIL,RDR);
      CZT   = CZ (ALPHA,BETA,EL);
%
      DAIL= AIL/20.0;
      DRDR= RDR/30.0;
      CLT = CL(ALPHA,BETA) + DLDA(ALPHA,BETA)*DAIL + DLDR(ALPHA,BETA)*DRDR;
      CMT = CM(ALPHA,EL);
      CNT = CN(ALPHA,BETA) + DNDA(ALPHA,BETA)*DAIL + DNDR(ALPHA,BETA)*DRDR;
%
%  ADD DAMPING DERIVATIVES :
%
      TVT   =  0.5/VT;
      B2V   =  B*TVT;
      CQ    =  CBAR*Q*TVT;
      D = DAMP(ALPHA);
%
      CXT=  CXT + CQ  *   D(1);
      CYT=  CYT + B2V * ( D(2)*R + D(3)*P );
      CZT=  CZT + CQ  *   D(4);
%
      CLT=  CLT + B2V * ( D(5)*R + D(6)*P );
      CMT=  CMT + CQ  *   D(7)        +       CZT * (XCGR-XCG);
      CNT=  CNT + B2V * ( D(8)*R + D(9)*P ) - CYT * (XCGR-XCG) * CBAR/B;

% GET READY FOR STATE EQUATIONS
%
%
      CBTA  = cos(X(3));
      U     = VT * cos(X(2)) * CBTA;
      V     = VT * sin(X(3));
      W     = VT * sin(X(2)) * CBTA;
%
      STH   = sin(THETA);
      CTH   = cos(THETA);
      SPH   = sin(PHI);
      CPH   = cos(PHI);
      SPSI  = sin(PSI);
      CPSI  = cos(PSI);
%
      QS    = QBAR * S;
      QSB   = QS * B;
      RMQS  = RM * QS;
      GCTH  = G * CTH;
      QSPH  = Q * SPH;
      AX    = RM * (QS * CXT + T);
      AY    = RMQS * CYT;
      AZ    = RMQS * CZT;
%
%  STATE EQUATIONS
%  FORCES.
      UDOT  =  R*V - Q*W - G*STH    +   AX;
      VDOT  =  P*W - R*U + GCTH * SPH + AY;
      WDOT  =  Q*U - P*V + GCTH * CPH + AZ;
      DUM   =  (U*U + W*W);
      XD(1) = (U*UDOT + V*VDOT + W*WDOT)/VT;
      XD(2) = (U*WDOT - W*UDOT) / DUM;
      XD(3) = (VT*VDOT- V*XD(1)) * CBTA / DUM;
%
%  KINEMATICS.
      XD(4) =  P + (STH/CTH)*(QSPH + R*CPH);
      XD(5) =       Q*CPH - R*SPH;
      XD(6) =      (QSPH + R*CPH)/CTH;
%
%  MOMENTS.
      XD(7) =  (C2*P + C1*R + C4*HE)*Q + QSB*(C3*CLT + C4*CNT);
      XD(8) =  (C5*P - C7*HE)*R + C6*(R*R-P*P) +QS*CBAR*C7*CMT;
      XD(9) =  (C8*P-C2*R+C9*HE)*Q + QSB*(C4*CLT + C9*CNT);
%
%  NAVIGATION
      T1= SPH * CPSI;
      T2= CPH * STH;
      T3= SPH * SPSI;
      S1= CTH * CPSI;
      S2= CTH * SPSI;
      S3= T1  * STH - CPH * SPSI;
      S4= T3  * STH + CPH * CPSI;
      S5= SPH * CTH;
      S6= T2  * CPSI + T3;
      S7= T2  * SPSI - T1;
      S8= CPH * CTH;
%
      XD(10)   =  U * S1 + V * S3  + W * S6;
      XD(11)   =  U * S2 + V * S4  + W * S7;
      XD(12)   = -(-U * STH + V * S5 + W * S8);
%
%  OUTPUTS :
      AZN     =  (-AZ + XA * XD(8) )/G;
      AYN   =   AY/G;
      AXN   =   AX/G;
      CSTAR= AZN + 12.4*Q;
      
      Y = [AZN;AYN;AXN;QBAR;MACH;CSTAR];
      
      return;
%
%
%
% ********************** LOOKUP TABLES FOR F16 *********************
%
% ****  ENGINE DATA *******************************
%
      function [rtauv] = RTAU(DP)
      if (DP<=25.0)
          rtauv=1.0;
      elseif (DP>=50.0)
          rtauv=0.1;
      else
          rtauv=1.9-.036*DP;
      end;
      return;
%
% **********************************
%
      function [tgearv] = TGEAR(THTL)
      if (THTL<=0.77) 
        tgearv = 64.94*THTL;
      else
        tgearv = 217.38*THTL-117.38;
      end
      return;
%
% *********************************
%
      function [pdotv] = PDOT(P3,P1)
      if (P1>=50.0)
        if (P3>=50.0)
          T=5.0;
          P2=P1;
        else
          P2=60.0;
          T=RTAU(P2-P3);
        end
      else
        if (P3>=50.0)
          T=5.0;
          P2=40.0;
        else
          P2=P1;
          T=RTAU(P2-P3);
        end
      end
      pdotv=T*(P2-P3);
      
      return
%
% ****************************************
%
      function [thrustv] = THRUST(POW,ALT,RMACH)

      persistent init A B C;
      
      if (isempty(init))
%        idle data now
      A = [1060.0,   670.0,   880.0,  1140.0,  1500.0,  1860.0; ...
            635.0,   425.0,   690.0,  1010.0,  1330.0,  1700.0; ... 
             60.0,    25.0,   345.0,   755.0,  1130.0,  1525.0; ...
          -1020.0,  -710.0,  -300.0,   350.0,   910.0,  1360.0; ...
          -2700.0, -1900.0, -1300.0,  -247.0,   600.0,  1100.0; ...
          -3600.0, -1400.0,  -595.0,  -342.0,  -200.0,   700.0]';
%         mil data now
      B=[12680.0,  9150.0,  6200.0,  3950.0,  2450.0,  1400.0; ...
         12680.0,  9150.0,  6313.0,  4040.0,  2470.0,  1400.0; ...
         12610.0,  9312.0,  6610.0,  4290.0,  2600.0,  1560.0; ...
         12640.0,  9839.0,  7090.0,  4660.0,  2840.0,  1660.0; ...
         12390.0, 10176.0,  7750.0,  5320.0,  3250.0,  1930.0; ...
         11680.0,  9848.0,  8050.0,  6100.0,  3800.0,  2310.0]';
%         max data now
      C=[20000.0, 15000.0, 10800.0,  7000.0,  4000.0,  2500.0; ...
         21420.0, 15700.0, 11225.0,  7323.0,  4435.0,  2600.0; ...
         22700.0, 16860.0, 12250.0,  8154.0,  5000.0,  2835.0; ...
         24240.0, 18910.0, 13760.0,  9285.0,  5700.0,  3215.0; ...
         26070.0, 21075.0, 15975.0, 11115.0,  6860.0,  3950.0; ...
         28886.0, 23319.0, 18300.0, 13484.0,  8642.0,  5057.0]';
         init=1;
      end;
%
         H = .0001*ALT;
         I = floor(H)+1;
         if (I>=6)
            I=5;
         end;
         DH= H-I+1;
         RM= 5.0*RMACH;
         M = floor(RM)+1;
         if (M>=6)
             M=5;
         end;
         DM= RM-M+1;
         CDH=1.0-DH;
         S= B(I,M)  *CDH + B(I+1,M)  *DH;
         T= B(I,M+1)*CDH + B(I+1,M+1)*DH;
         TMIL= S + (T-S)*DM;
         if( POW < 50.0 )
            S= A(I,M)  *CDH + A(I+1,M)  *DH;
            T= A(I,M+1)*CDH + A(I+1,M+1)*DH;
            TIDL= S + (T-S)*DM;
            thrustv=TIDL+(TMIL-TIDL)*POW*.02;
         else
            S= C(I,M)  *CDH + C(I+1,M)  *DH;
            T= C(I,M+1)*CDH + C(I+1,M+1)*DH;
            TMAX= S + (T-S)*DM;
            thrustv=TMIL+(TMAX-TMIL)*(POW-50.0)*.02;
         end;
      return;
      
%
%    ****** FORCES *****
%
      function [cxv] = CX(ALPHA,EL)
      persistent init A;
      
      if (isempty(init))
        A=[-.099, -.081, -.081, -.063, -.025,  .044,  .097, ...
            .113,  .145,  .167,  .174,  .166; ...
           -.048, -.038, -.040, -.021,  .016,  .083,  .127, ...
            .137,  .162,  .177,  .179,  .167; ...
           -.022, -.020, -.021, -.004,  .032,  .094,  .128, ...
            .130,  .154,  .161,  .155,  .138; ...
           -.040, -.038, -.039, -.025,  .006,  .062,  .087, ...
            .085,  .100,  .110,  .104,  .091; ...
           -.083, -.073, -.076, -.072, -.046,  .012,  .024, ...
            .025,  .043,  .053,  .047,  .040]';
        init=1;
      end;
%
%
      S= 0.2 * ALPHA;
      K= floor(S)+3;
      if (K <=1) K= 2; end;
      if (K >=12) K=11; end;
      DA= S - K+3;
      L = K + fix(1.1*sign(DA));
%
      S= EL/12.0;
      M= floor(S)+3;
      if(M <= 1) M= 2; end;
      if(M >=  5) M=  4; end;
      DE= S - M+3;
      N= M + fix(1.1*sign(DE));
%
      T= A(K,M);
      U= A(K,N);
      V= T + abs(DA) * (A(L,M) - T);
      W= U + abs(DA) * (A(L,N) - U);
      cxv = V + (W-V)  * abs(DE);
      return;
%
%
      function [cyv] = CY(BETA,AIL,RDR);
%
      cyv = -.02*BETA + .021*(AIL/20.0) + .086*(RDR/30.0);
      
      return;
%
      function [czv] = CZ(ALPHA,BETA,EL);
      persistent init A;
      
      if (isempty(init))
        A= [.770,   .241,  -.100,  -.416,  -.731, -1.053, -1.366, ...
          -1.646, -1.917, -2.120, -2.248, -2.229];
        init=1;
      end;
%
      S= 0.2 * ALPHA;
      K= floor(S)+3;
      if(K <= 1) K= 2; end;
      if(K >=  12) K=  11; end;
      DA= S - K + 3;
      L = K + fix( 1.1*sign(DA) );
      S= A(K) + abs(DA) * (A(L) - A(K));
      czv = S*(1-(BETA/57.3)^2) - .19*(EL/25.0);
      return;
%
% ******* MOMENTS *************
%
      function [cmv] = CM(ALPHA,EL);
      persistent init A;
      
      if (isempty(init))
        A=[.205,  .168,  .186,  .196,  .213,  .251,  .245, ...
           .238,  .252,  .231,  .198,  .192; ...
           .081,  .077,  .107,  .110,  .110,  .141,  .127, ...
           .119,  .133,  .108,  .081,  .093; ...
          -.046, -.020, -.009, -.005, -.006,  .010,  .006, ...
          -.001,  .014,  .000, -.013,  .032; ...
          -.174, -.145, -.121, -.127, -.129, -.102, -.097, ...
          -.113, -.087, -.084, -.069, -.006; ...
          -.259, -.202, -.184, -.193, -.199, -.150, -.160, ...
          -.167, -.104, -.076, -.041, -.005]';
        init=1;
      end;
%
%
      S= 0.2 * ALPHA;
      K= floor(S)+3;
      if(K <= 1) K= 2; end;
      if(K >= 12) K=  11; end;
      DA= S - K +3;
      L = K + fix( 1.1*sign(DA) );
%
      S= EL/12.0;
      M= floor(S)+3;
      if(M <= 1) M= 2; end;
      if(M >=  5) M=  4; end;
      DE= S - M +3;
      N= M + fix( 1.1*sign(DE) );
%
      T= A(K,M);
      U= A(K,N);
      V= T + abs(DA) * (A(L,M) - T);
      W= U + abs(DA) * (A(L,N) - U);
      cmv = V + (W-V)  * abs(DE);
      return;
%
      function [clv] = CL(ALPHA,BETA);
      persistent init A;
      
      if (isempty(init))
        A=[   0.,     0.,     0.,     0.,     0.,     0.,     0., ...
              0.,     0.,     0.,     0.,     0.; ...     
           -.001,  -.004,  -.008,  -.012,  -.016,  -.019,  -.020, ...
           -.020,  -.015,  -.008,  -.013,  -.015; ...
           -.003,  -.009,  -.017,  -.024,  -.030,  -.034,  -.040, ...
           -.037,  -.016,  -.002,  -.010,  -.019; ...
           -.001,  -.010,  -.020,  -.030,  -.039,  -.044,  -.050, ...
           -.049,  -.023,  -.006,  -.014,  -.027; ...
            .000,  -.010,  -.022,  -.034,  -.047,  -.046,  -.059, ...
           -.061,  -.033,  -.036,  -.035,  -.035; ...
            .007,  -.010,  -.023,  -.034,  -.049,  -.046,  -.068, ...
           -.071,  -.060,  -.058,  -.062,  -.059; ...
            .009,  -.011,  -.023,  -.037,  -.050,  -.047,  -.074, ...
           -.079,  -.091,  -.076,  -.077,  -.076]';
        init=1;
      end;
%
      S= 0.2 * ALPHA;
      K= floor(S)+3;
      if(K <= 1) K= 2; end;
      if(K >=  12) K=  11; end;
      DA= S - K +3;
      L = K + fix( 1.1*sign(DA) );
%
      S= .2* abs(BETA);
      M= floor(S)+1;
      if(M <=  1) M= 2; end;
      if(M >=  7) M= 6; end;
      DB= S - M +1;
      N= M + fix( 1.1*sign(DB) );
%
      T= A(K,M);
      U= A(K,N);
      V= T + abs(DA) * (A(L,M) - T);
      W= U + abs(DA) * (A(L,N) - U);
      DUM= V + (W-V)  * abs(DB);
      clv = DUM * sign(BETA);
      return;
%
%
      function [cnv] = CN(ALPHA,BETA);

      persistent init A; 
      
      if (isempty(init))
        A=[   0.,     0.,     0.,     0.,     0.,     0.,     0., ...
              0.,     0.,     0.,     0.,     0.; ...
            .018,   .019,   .018,   .019,   .019,   .018,   .013, ...
            .007,   .004,  -.014,  -.017,  -.033; ...
            .038,   .042,   .042,   .042,   .043,   .039,   .030, ...
            .017,   .004,  -.035,  -.047,  -.057; ...
            .056,   .057,   .059,   .058,   .058,   .053,   .032, ...
            .012,   .002,  -.046,  -.071,  -.073; ...
            .064,   .077,   .076,   .074,   .073,   .057,   .029, ...
            .007,   .012,  -.034,  -.065,  -.041; ...
            .074,   .086,   .093,   .089,   .080,   .062,   .049, ...
            .022,   .028,  -.012,  -.002,  -.013; ...
            .079,   .090,   .106,   .106,   .096,   .080,   .068, ...
            .030,   .064,   .015,   .011,  -.001]';
        init=1;
      end;
%
%
      S= 0.2 * ALPHA;
      K= floor(S)+3;
      if(K <= 1) K= 2; end;
      if(K >=  12) K=  11; end;
      DA= S - K +3;
      L = K + fix( 1.1*sign(DA) );
%
      S= .2* abs(BETA);
      M= floor(S) + 1;
      if(M <=  1) M= 2; end;
      if(M >=  7) M= 6; end;
      DB= S - M +1;
      N= M + fix( 1.1*sign(DB) );
%
      T= A(K,M);
      U= A(K,N);
      V= T + abs(DA) * (A(L,M) - T);
      W= U + abs(DA) * (A(L,N) - U);
      DUM= V + (W-V)  * abs(DB);
      cnv = DUM * sign(BETA);
      return;
      
%
% *********** CONTROL EFFECTS *************
%
      function [dldav] = DLDA(ALPHA,BETA);
      persistent init A;
      
      if (isempty(init))
        A=[-.041,  -.052,  -.053,  -.056,  -.050,  -.056,  -.082, ...
           -.059,  -.042,  -.038,  -.027,  -.017; ...
           -.041,  -.053,  -.053,  -.053,  -.050,  -.051,  -.066, ...
           -.043,  -.038,  -.027,  -.023,  -.016; ...
           -.042,  -.053,  -.052,  -.051,  -.049,  -.049,  -.043, ...
           -.035,  -.026,  -.016,  -.018,  -.014; ...
           -.040,  -.052,  -.051,  -.052,  -.048,  -.048,  -.042, ...
           -.037,  -.031,  -.026,  -.017,  -.012; ...
           -.043,  -.049,  -.048,  -.049,  -.043,  -.042,  -.042, ...
           -.036,  -.025,  -.021,  -.016,  -.011; ...
           -.044,  -.048,  -.048,  -.047,  -.042,  -.041,  -.020, ...
           -.028,  -.013,  -.014,  -.011,  -.010; ...
           -.043,  -.049,  -.047,  -.045,  -.042,  -.037,  -.003, ...
           -.013,  -.010,  -.003,  -.007,  -.008]';
        init=1;
      end;
%
%
      S= 0.2 * ALPHA;
      K= floor(S)+3;
      if(K <= 1) K= 2; end;
      if(K >=  12) K=  11; end;
      DA= S - K +3;
      L = K + fix( 1.1*sign(DA) );
%
      S= 0.1 * BETA;
      M= floor(S)+4;
      if(M <= 1) M= 2; end;
      if(M >=  7) M=  6; end;
      DB= S - M + 4;
      N= M + fix( 1.1*sign(DB) );
%
      T= A(K,M);
      U= A(K,N);
      V= T + abs(DA) * (A(L,M) - T);
      W= U + abs(DA) * (A(L,N) - U);
      dldav = V + (W-V)  * abs(DB);
      return;
      
%
%
      function [dldrv] = DLDR(ALPHA,BETA);
      persistent init A;
      
      if (isempty(init))
        A=[.005,   .017,   .014,   .010,  -.005,   .009,   .019, ...
           .005,  -.000,  -.005,  -.011,   .008; ...
           .007,   .016,   .014,   .014,   .013,   .009,   .012, ...
           .005,   .000,   .004,   .009,   .007; ...
           .013,   .013,   .011,   .012,   .011,   .009,   .008, ...
           .005,  -.002,   .005,   .003,   .005; ...
           .018,   .015,   .015,   .014,   .014,   .014,   .014, ...
           .015,   .013,   .011,   .006,   .001; ...
           .015,   .014,   .013,   .013,   .012,   .011,   .011, ...
           .010,   .008,   .008,   .007,   .003; ...
           .021,   .011,   .010,   .011,   .010,   .009,   .008, ...
           .010,   .006,   .005,   .000,   .001; ...
           .023,   .010,   .011,   .011,   .011,   .010,   .008, ...
           .010,   .006,   .014,   .020,   .000]';
        init=1;
      end;
%
%
      S= 0.2 * ALPHA;
      K= floor(S)+3;
      if(K <= 1) K= 2; end;
      if(K >=  12) K=  11; end;
      DA= S - K +3;
      L = K + fix( 1.1*sign(DA) );
%
      S= 0.1 * BETA;
      M= floor(S)+4;
      if(M <= 1) M= 2; end;
      if(M >=  7) M=  6; end;
      DB= S - M + 4;
      N= M + fix( 1.1*sign(DB) );
%
      T= A(K,M);
      U= A(K,N);
      V= T + abs(DA) * (A(L,M) - T);
      W= U + abs(DA) * (A(L,N) - U);
      dldrv = V + (W-V)  * abs(DB);
      return;
%
%
      function [dndav] = DNDA(ALPHA,BETA);
      persistent init A;
      
      if (isempty(init))
        A=[.001,  -.027,  -.017,  -.013,  -.012,  -.016,   .001, ...
           .017,   .011,   .017,   .008,   .016; ...
           .002,  -.014,  -.016,  -.016,  -.014,  -.019,  -.021, ...
           .002,   .012,   .016,   .015,   .011; ...
          -.006,  -.008,  -.006,  -.006,  -.005,  -.008,  -.005, ...
           .007,   .004,   .007,   .006,   .006; ...
          -.011,  -.011,  -.010,  -.009,  -.008,  -.006,   .000, ...
           .004,   .007,   .010,   .004,   .010; ...
          -.015,  -.015,  -.014,  -.012,  -.011,  -.008,  -.002, ...
           .002,   .006,   .012,   .011,   .011; ...
          -.024,  -.010,  -.004,  -.002,  -.001,   .003,   .014, ...
           .006,  -.001,   .004,   .004,   .006; ...
          -.022,   .002,  -.003,  -.005,  -.003,  -.001,  -.009, ...
          -.009,  -.001,   .003,  -.002,   .001]';
        init=1;
      end;
%
%
      S= 0.2 * ALPHA;
      K= floor(S)+3;
      if(K <= 1) K= 2; end;
      if(K >=  12) K=  11; end;
      DA= S - K +3;
      L = K + fix( 1.1*sign(DA) );
%
      S= 0.1 * BETA;
      M= floor(S)+4;
      if(M <= 1) M= 2; end;
      if(M >=  7) M=  6; end;
      DB= S - M + 4;
      N= M + fix( 1.1*sign(DB) );
%
      T= A(K,M);
      U= A(K,N);
      V= T + abs(DA) * (A(L,M) - T);
      W= U + abs(DA) * (A(L,N) - U);
      dndav = V + (W-V)  * abs(DB);
      return;
%
%
      function [dndrv] = DNDR(ALPHA,BETA);
      persistent init A;
      
      if (isempty(init))
        A=[-.018,  -.052,  -.052,  -.052,  -.054,  -.049,  -.059, ...
          -.051,  -.030,  -.037,  -.026,  -.013; ...
          -.028,  -.051,  -.043,  -.046,  -.045,  -.049,  -.057, ...
          -.052,  -.030,  -.033,  -.030,  -.008; ...
          -.037,  -.041,  -.038,  -.040,  -.040,  -.038,  -.037, ...
          -.030,  -.027,  -.024,  -.019,  -.013; ...
          -.048,  -.045,  -.045,  -.045,  -.044,  -.045,  -.047, ...
          -.048,  -.049,  -.045,  -.033,  -.016; ...
          -.043,  -.044,  -.041,  -.041,  -.040,  -.038,  -.034, ...
          -.035,  -.035,  -.029,  -.022,  -.009; ...
          -.052,  -.034,  -.036,  -.036,  -.035,  -.028,  -.024, ...
          -.023,  -.020,  -.016,  -.010,  -.014; ...
          -.062,  -.034,  -.027,  -.028,  -.027,  -.027,  -.023, ...
          -.023,  -.019,  -.009,  -.025,  -.010]';
        init=1;
      end;
%
%
      S= 0.2 * ALPHA;
      K= floor(S)+3;
      if(K <= 1) K= 2; end;
      if(K >=  12) K=  11; end;
      DA= S - K +3;
      L = K + fix( 1.1*sign(DA) );
%
      S= 0.1 * BETA;
      M= floor(S)+4;
      if(M <= 1) M= 2; end;
      if(M >=  7) M=  6; end;
      DB= S - M + 4;
      N= M + fix( 1.1*sign(DB) );
%
      T= A(K,M);
      U= A(K,N);
      V= T + abs(DA) * (A(L,M) - T);
      W= U + abs(DA) * (A(L,N) - U);
      dndrv = V + (W-V)  * abs(DB);
      return;
%
% ********** AERODYNAMIC DAMPING ***********
%
      function [D] = DAMP(ALPHA);
      persistent init A;
      
      if (isempty(init))
        A=[-.267,  -.110,   .308,   1.34,   2.08,   2.91,   2.76, ...
            2.05,   1.50,   1.49,   1.83,   1.21; ...
            .882,   .852,   .876,   .958,   .962,   .974,   .819, ...
            .483,   .590,   1.21,  -.493,  -1.04; ...
           -.108,  -.108,  -.188,   .110,   .258,   .226,   .344, ...
            .362,   .611,   .529,   .298,  -2.27; ...
           -8.80,  -25.8,  -28.9,  -31.4,  -31.2,  -30.7,  -27.7, ...
           -28.2,  -29.0,  -29.8,  -38.3,  -35.3; ...
           -.126,  -.026,   .063,   .113,   .208,   .230,   .319, ...
            .437,   .680,   .100,   .447,  -.330; ...
           -.360,  -.359,  -.443,  -.420,  -.383,  -.375,  -.329, ...
           -.294,  -.230,  -.210,  -.120,  -.100; ...
           -7.21,  -.540,  -5.23,  -5.26,  -6.11,  -6.64,  -5.69, ...
           -6.00,  -6.20,  -6.40,  -6.60,  -6.00; ...
           -.380,  -.363,  -.378,  -.386,  -.370,  -.453,  -.550, ...
           -.582,  -.595,  -.637,  -1.02,  -.840; ...
            .061,   .052,   .052,  -.012,  -.013,  -.024,   .050, ...
            .150,   .130,   .158,   .240,   .150]';
        init=1;
      end;
%
%
      S= 0.2 * ALPHA;
      K= floor(S)+3;
      if(K <= 1) K= 2; end;
      if(K >=  12) K=  11; end;
      DA= S - K +3;
      L = K + fix( 1.1*sign(DA) );
%
      D=zeros(9,1);
      for I= 1:9;
        D(I)= A(K,I) + abs(DA) * (A(L,I) - A(K,I));
      end;
      return;