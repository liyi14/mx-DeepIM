im = imread('bb8.jpg');
% imshow(im);
[Y, X, ~] = size(im);
p0_0 = [1115, 200]; % (y0,x0)
dx = 28.6; % 1px
dy = 10.77;
% % abs(p - p0_0)./[dy, dx]
ape_yxs = [41,  1588; 
           119, 652; 
           237, 491;
           369, 428;
           975, 316;
           1065, 286;
           64, 1042;
           41, 1590;
           1111, 202;
           ];
ape_yxs = sortrows(ape_yxs, 1);

ape_yxs = bsxfun(@rdivide, abs(bsxfun(@minus, ape_yxs, p0_0)), [dy, dx]);
plot(ape_yxs(:,2), ape_yxs(:,1), 'b-');

% can
can_yxs = [42, 1588;
    45, 968;
    58, 859;
    106, 742;
   168,  698; 
   244, 661; 
   417, 601;
   510, 571;
   606, 542;
   820, 482;
   1011, 410;
   1071, 379;
   1099, 353;
   1111, 202;
   ];
can_yxs = sortrows(can_yxs, 1);

can_yxs = bsxfun(@rdivide, abs(bsxfun(@minus, can_yxs, p0_0)), [dy, dx]);
hold on;
grid on;
plot(can_yxs(:,2), can_yxs(:,1), 'g-');

% cat
cat_yxs = [42, 1588;
    57, 1038;
    80, 828;
    100, 714;
    126, 683;
    155, 627;
    214, 573;
    349, 513;
    461, 481;
    678, 426;
   898, 369;
   1085, 318;
   1103, 298;
   1111, 202;
   ];
cat_yxs = sortrows(cat_yxs, 1);

cat_yxs = bsxfun(@rdivide, abs(bsxfun(@minus, cat_yxs, p0_0)), [dy, dx]);
hold on;
grid on;
plot(cat_yxs(:,2), cat_yxs(:,1), 'r-');

% driller
driller_yxs = [42, 1588;
    40, 1196;
    52, 987;
    63, 912;
    91, 852;
    110, 796;
    176, 712;
    201, 694;
    232, 655;
    308, 616;
    483, 554;
    592, 529;
    694, 503;
    851, 466;
   1043, 405;
   1094, 372;
   1109, 345;
   1111, 202;
   ];
driller_yxs = sortrows(driller_yxs, 1);

driller_yxs = bsxfun(@rdivide, abs(bsxfun(@minus, driller_yxs, p0_0)), [dy, dx]);
hold on;
grid on;
plot(driller_yxs(:,2), driller_yxs(:,1),'linestyle', '-', 'color', [25, 193, 193]./255);

% duck
duck_yxs = [42, 1588;
    40, 1586;
    43, 1384;
    48, 1225;
    77, 1039;
    77, 1003;
    99, 942;
    200, 799;
    290, 730;
    403, 638;
    479, 577;
    527, 545;
    694, 476;
    916, 398;
   995, 367;
   1044, 342;
   1102, 291;
   1111, 202;
   ];
duck_yxs = sortrows(duck_yxs, 1);

duck_yxs = bsxfun(@rdivide, abs(bsxfun(@minus, duck_yxs, p0_0)), [dy, dx]);
hold on;
grid on;
plot(duck_yxs(:,2), duck_yxs(:,1),'linestyle', '-', 'color', [184, 24, 186]./255);

% glue
glue_yxs = [
    48, 1592;
    65, 1451;
    89, 1265;
    107, 1235;
    150, 1049;
    155, 996;
    167, 941;
    195, 890;
    210, 826;
    265, 750;
    304, 687;
    420, 613;
    518, 574;
    594, 536;
    679, 498;
    836, 437;
   939, 394;
   1010, 367;
   1106, 315;
   1111, 202;
   ];
glue_yxs = sortrows(glue_yxs, 1);

glue_yxs = bsxfun(@rdivide, abs(bsxfun(@minus, glue_yxs, p0_0)), [dy, dx]);
hold on;
grid on;
plot(glue_yxs(:,2), glue_yxs(:,1),'linestyle', '-', 'color', [193, 192, 40]./255);

% holepuncher
holepuncher_yxs = [
    60, 1590;
    63, 1448;
    62, 1451;
    87, 1326;
    204, 903;
    263, 741;
    439, 572;
    701, 464;
   1024, 378;
   1090, 349;
   1110, 315;
   1111, 202;
   ];
holepuncher_yxs = sortrows(holepuncher_yxs, 1);

holepuncher_yxs = bsxfun(@rdivide, abs(bsxfun(@minus, holepuncher_yxs, p0_0)), [dy, dx]);
hold on;
grid on;
plot(holepuncher_yxs(:,2), holepuncher_yxs(:,1),'linestyle', '-', 'color', [0, 0, 0]./255);

legend('Ape', 'Can', 'Cat', 'Driller', 'Duck', 'Glue', 'Holepuncher', 'Location', 'best');

cur_dir = pwd;
res_path = [cur_dir, '/bb8_yxs.mat'];
res = cell(7, 2);
res(1, :) = {'ape', ape_yxs};
res(2, :) = {'can', can_yxs};
res(3, :) = {'cat', cat_yxs};
res(4, :) = {'driller', driller_yxs};
res(5, :) = {'duck', duck_yxs};
res(6, :) = {'glue', glue_yxs};
res(7, :) = {'holepuncher', holepuncher_yxs};
save(res_path, 'res');
fprintf(['results saved to: ', res_path, '\n']);

       
       

