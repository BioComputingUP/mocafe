Search.setIndex({docnames:["bib","code_documentation","demo_doc/angiogenesis_2d","demo_doc/angiogenesis_3d","demo_doc/index","demo_doc/introduction_to_fenics","demo_doc/prostate_cancer2d","demo_doc/prostate_cancer3d","index","installation","sub_code_doc/angie","sub_code_doc/angie_sub_code_doc/af_sourcing","sub_code_doc/angie_sub_code_doc/base_classes","sub_code_doc/angie_sub_code_doc/forms","sub_code_doc/angie_sub_code_doc/tipcells","sub_code_doc/expressions","sub_code_doc/fenut","sub_code_doc/fenut_sub_code_doc/fenut","sub_code_doc/fenut_sub_code_doc/log","sub_code_doc/fenut_sub_code_doc/mansimdata","sub_code_doc/fenut_sub_code_doc/parameters","sub_code_doc/fenut_sub_code_doc/solvers","sub_code_doc/litforms","sub_code_doc/litforms_sub_code_doc/prostate_cancer","sub_code_doc/litforms_sub_code_doc/xu16","sub_code_doc/math"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinxcontrib.bibtex":9,sphinx:56},filenames:["bib.rst","code_documentation.rst","demo_doc/angiogenesis_2d.rst","demo_doc/angiogenesis_3d.rst","demo_doc/index.rst","demo_doc/introduction_to_fenics.rst","demo_doc/prostate_cancer2d.rst","demo_doc/prostate_cancer3d.rst","index.rst","installation.rst","sub_code_doc/angie.rst","sub_code_doc/angie_sub_code_doc/af_sourcing.rst","sub_code_doc/angie_sub_code_doc/base_classes.rst","sub_code_doc/angie_sub_code_doc/forms.rst","sub_code_doc/angie_sub_code_doc/tipcells.rst","sub_code_doc/expressions.rst","sub_code_doc/fenut.rst","sub_code_doc/fenut_sub_code_doc/fenut.rst","sub_code_doc/fenut_sub_code_doc/log.rst","sub_code_doc/fenut_sub_code_doc/mansimdata.rst","sub_code_doc/fenut_sub_code_doc/parameters.rst","sub_code_doc/fenut_sub_code_doc/solvers.rst","sub_code_doc/litforms.rst","sub_code_doc/litforms_sub_code_doc/prostate_cancer.rst","sub_code_doc/litforms_sub_code_doc/xu16.rst","sub_code_doc/math.rst"],objects:{"mocafe.angie":[[11,0,0,"-","af_sourcing"],[12,0,0,"-","base_classes"],[13,0,0,"-","forms"],[14,0,0,"-","tipcells"]],"mocafe.angie.af_sourcing":[[11,1,1,"","ClockChecker"],[11,1,1,"","ConstantSourcesField"],[11,1,1,"","RandomSourceMap"],[11,1,1,"","SourceCell"],[11,1,1,"","SourceMap"],[11,1,1,"","SourcesManager"],[11,3,1,"","sources_in_circle_points"]],"mocafe.angie.af_sourcing.ClockChecker":[[11,2,1,"","clock_check"]],"mocafe.angie.af_sourcing.SourceMap":[[11,2,1,"","get_global_source_cells"],[11,2,1,"","get_local_source_cells"],[11,2,1,"","remove_global_source"]],"mocafe.angie.af_sourcing.SourcesManager":[[11,2,1,"","apply_sources"],[11,2,1,"","remove_sources_near_vessels"]],"mocafe.angie.base_classes":[[12,3,1,"","fibonacci_sphere"]],"mocafe.angie.forms":[[13,3,1,"","angiogenesis_form"],[13,3,1,"","angiogenic_factor_form"],[13,3,1,"","cahn_hillard_form"],[13,3,1,"","vascular_proliferation_form"]],"mocafe.angie.tipcells":[[14,1,1,"","TipCell"],[14,1,1,"","TipCellManager"],[14,1,1,"","TipCellsField"]],"mocafe.angie.tipcells.TipCell":[[14,2,1,"","get_radius"],[14,2,1,"","is_point_inside"],[14,2,1,"","move"]],"mocafe.angie.tipcells.TipCellManager":[[14,2,1,"","activate_tip_cell"],[14,2,1,"","compute_tip_cell_velocity"],[14,2,1,"","get_global_tip_cells_list"],[14,2,1,"","move_tip_cells"],[14,2,1,"","revert_tip_cells"]],"mocafe.angie.tipcells.TipCellsField":[[14,2,1,"","add_tip_cell"],[14,2,1,"","compute_phi_c"],[14,2,1,"","eval"]],"mocafe.expressions":[[15,1,1,"","EllipseField"],[15,1,1,"","EllipsoidField"],[15,1,1,"","PythonFunctionField"],[15,1,1,"","SmoothCircle"],[15,1,1,"","SmoothCircularTumor"]],"mocafe.fenut":[[17,0,0,"-","fenut"],[18,0,0,"-","log"],[19,0,0,"-","mansimdata"],[20,0,0,"-","parameters"],[21,0,0,"-","solvers"]],"mocafe.fenut.fenut":[[17,3,1,"","build_local_box"],[17,3,1,"","divide_in_chunks"],[17,3,1,"","flatten_list_of_lists"],[17,3,1,"","get_mixed_function_space"],[17,3,1,"","is_in_local_box"],[17,3,1,"","is_point_inside_mesh"],[17,3,1,"","load_parameters"],[17,3,1,"","setup_pvd_files"],[17,3,1,"","setup_xdmf_files"]],"mocafe.fenut.log":[[18,1,1,"","DebugAdapter"],[18,1,1,"","InfoCsvAdapter"],[18,3,1,"","confgure_root_logger_with_standard_settings"]],"mocafe.fenut.log.DebugAdapter":[[18,2,1,"","process"]],"mocafe.fenut.log.InfoCsvAdapter":[[18,2,1,"","process"]],"mocafe.fenut.mansimdata":[[19,3,1,"","save_sim_info"],[19,3,1,"","setup_data_folder"]],"mocafe.fenut.parameters":[[20,1,1,"","Parameters"],[20,3,1,"","from_dict"],[20,3,1,"","from_ods_sheet"]],"mocafe.fenut.parameters.Parameters":[[20,2,1,"","as_dataframe"],[20,2,1,"","get_value"],[20,2,1,"","is_parameter"],[20,2,1,"","is_value_present"],[20,2,1,"","set_value"]],"mocafe.fenut.solvers":[[21,1,1,"","PETScNewtonSolver"],[21,1,1,"","PETScProblem"]],"mocafe.fenut.solvers.PETScNewtonSolver":[[21,2,1,"","solver_setup"]],"mocafe.fenut.solvers.PETScProblem":[[21,2,1,"","F"],[21,2,1,"","J"]],"mocafe.litforms":[[23,0,0,"-","prostate_cancer"],[24,0,0,"-","xu16"]],"mocafe.litforms.prostate_cancer":[[23,3,1,"","df_dphi"],[23,3,1,"","prostate_cancer_chem_potential"],[23,3,1,"","prostate_cancer_form"],[23,3,1,"","prostate_cancer_nutrient_form"]],"mocafe.litforms.xu16":[[24,3,1,"","xu2016_nutrient_form"],[24,3,1,"","xu_2016_cancer_form"]],"mocafe.math":[[25,3,1,"","estimate_cancer_area"],[25,3,1,"","estimate_capillaries_area"],[25,3,1,"","shf"],[25,3,1,"","sigmoid"]],mocafe:[[15,0,0,"-","expressions"],[25,0,0,"-","math"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function"},terms:{"0":[0,2,3,5,6,7,9,13,14,15,24,25],"000":[2,3,5,6,7],"0000":19,"0001":19,"001":[5,6,7],"0019989":0,"009":0,"01":[6,7],"0149422":0,"03":0,"0e6":[6,7],"1":[0,2,3,5,6,7,12,13,14,15,17,24,25],"10":[0,9],"100":[0,2,3,6,7,15,25],"1000":[6,7],"1003":[6,7],"1007":0,"1016":0,"1058693490":0,"1073":0,"11":[0,2,3,10,11,13,14],"1145":0,"11588":0,"130":[6,7],"1371":0,"14":[0,1,5],"15":[0,5,8],"150":[6,7],"16":[0,6,7,23],"1615791113":0,"17":[0,6],"2":[0,1,2,3,5,6,7,13,14,17],"200":3,"2000":[6,7],"2011":[0,2,3,13,14],"2013":0,"2014":0,"2015":0,"2016":[0,6,7],"2017":0,"2019":[9,17],"2021":0,"20553":0,"225":[],"2368":[],"239":[],"25":5,"2566630":0,"27s_laws_of_diffus":0,"2d":[2,3,5,6,7,14],"3":[0,1,2,3,9,14,17],"30":5,"300":2,"319":0,"32":5,"33287":[],"365":6,"37":2,"375":2,"3d":[4,5,14],"4":[2,3,6,7,9,13,14,17],"4th":13,"5":[0,2,5,6,7,17,24],"500":7,"512":[],"515":0,"52461":0,"52462":0,"548":0,"6":[0,17,24],"600":[6,7],"642":[],"6_9":[],"6e5":[6,7],"7":[0,17],"73":[6,7],"75":[6,7],"8":[3,17],"9":0,"961":[6,7],"978":0,"\u00e1":0,"\u00e9":0,"\u00f8lgaard":0,"aln\u00e6":0,"boolean":2,"case":[2,3,5,6,7,9],"class":[1,2,5,6,11,12,14,15,16,17,18,20,21],"default":[2,5,6,9,10,17,19,25],"do":[2,3,5,6,7,9,14],"final":[2,5,6],"float":[15,17,19,25],"function":[1,2,5,6,7,11,13,14,15,17,23,24,25],"hern\u00e1ndez":[],"import":[2,3,5,6,7,9],"int":[2,3,11,14,17,21],"long":17,"new":[2,5,6,7,9,14,19,20],"poir\u00e9":[],"public":[23,24],"return":[2,11,12,13,14,17,18,19,20,23,24,25],"short":[2,3,6,7],"true":[2,3,7,11,14,17,19,20],"try":9,"var":9,"while":[2,3,6],A:[0,2,4,6,7,9,11,17,23],And:[2,5,6,7,9],As:[2,3,5,6,7],At:[],By:10,For:[2,3,5,6,19,23,24],If:[5,6,9,11,13,14,17,19],In:[1,2,3,5,6,7,8,9,13,14,17,25],It:[5,9,11,20,23],ONE:0,Of:[3,6,7,9],One:[2,3,5,7],The:[0,1,2,3,6,7,8,9,11,13,14,15,17,18,19,20,23,24],Then:[2,3,5,6,7,9],There:9,These:[2,6],To:[2,3,5,6,9],Will:19,With:[2,3,6,7],_:[],__file__:[2,3,6,7],__name__:[3,6,7],abh:[0,5,8],abl:[3,5,6,7],about:[5,20],abov:[2,3,5,6,7,9,14],academi:0,access:[0,5,11,22],accord:[2,5,6,14,24],accordingli:[2,3,6,7],account:2,acm:0,across:[2,5],act:11,activ:[2,3,14],activate_tip_cel:[2,3,14],actual:[2,3,5,6],ad:[2,3,13,19],adapt:18,add:[9,14],add_tip_cel:14,addit:2,addition:2,additionali:6,adimdiment:[6,7],adimension:[6,7],adiment:[6,7],advanc:5,advantag:6,af:[2,3,11,13,14],af_0:[2,3,13],af_:2,af_at_point:14,af_c:2,af_expression_funct:[],af_p:[2,13,14],af_sourc:[1,2,3,10],afexpressionfunct:[],after:[2,3,6],again:[2,3,5,7],agent:10,agreement:6,aim:[],al:[2,3,6,11,13,14,24],algebra:[5,6,21],algorithm:[2,5,10,12],all:[2,3,4,5,6,9,10,14,23],allow:[],almost:17,alnaeslolgaard:[0,1,5],along:[2,6],alpha:13,alpha_p:[2,13,14],alpha_t:[2,13],alreadi:[2,3,6,9],also:[2,3,4,5,6,8,9,11,14,23,24],altern:9,alwai:[3,7],amg:3,among:[],an:[0,2,3,5,6,7,8,9,10,11,15,19,20],analyz:6,ander:0,angi:[1,2,3,8],angiogen:[2,11,13,14],angiogenesi:[0,1,4,8,10,11,13,14],angiogenesis_2d:2,angiogenesis_2d_initial_condit:[],angiogenesis_3d:3,angiogenesis_form:[2,3,13],angiogenic_factor_form:[2,3,13],ani:[2,3,5,6,7,9,11,20],anim:5,anoth:2,anymor:9,apoptosi:[6,23],append:[],appli:[0,2,3,6,11,14],applic:0,apply_sourc:[2,3,11],appreci:[],approach:[9,10],appropri:[3,7],approxim:[2,5],apt:[],ar:[2,3,5,6,7,9,10,11,13,14,17,19,20],archiv:0,area:25,arg0:21,arg1:21,arg2:21,arg3:21,arg:18,argument:[2,5,6,13,18,19,23,24],argv:6,around:[],arrai:[6,7],articl:0,as_datafram:20,ask:[2,19],asm:[],aspect:16,assign:[2,3,5,6,7],associ:23,assum:6,august:0,author:6,auto:19,auto_enumer:[2,3,6,7,19],automat:[2,6,9,19],autoremov:9,auxiliari:[2,3,13],auxilliari:2,avail:9,averag:6,avoid:6,b:[0,2,3,6,7,17,25],backend:[5,21],backward:[5,23,24],ban:[],bar:[2,3,6,7],base:[6,8,12,24],base_class:[1,10],base_loc:[],basecel:11,bash_complet:9,basi:[13,15],basic:20,bc:21,becaus:[2,5,6,7],becom:[2,9],been:[3,9,11],befor:[2,3,5,6],begin:[5,6],behav:2,behaviour:[2,5,9],being:9,below:[1,2,5,9,13,17],bengzon:[],benjamin:0,berlin:[],best:[5,6,7,9,17],better:[6,8],between:[6,25],beyond:[],bibliographi:8,bigger:17,bin:9,bind:9,bit:[2,6],black:5,blechta:0,block:[],blood:[3,11],book:5,bool:19,border:[15,17],border_width:17,bot:[],both:[2,3,5,6,7,9,18],box:[5,9,17],boxmesh:[3,7],brief:4,briefli:[2,5,13],bug:9,build:[9,17,23,24],build_local_box:17,built:[5,6],builtin:5,c0:13,c:[2,3,5,6,11,13,14,24,25],c_0:[2,3],c_0_exp:2,c_c:2,c_exp:3,c_x:5,c_y:5,cach:9,cahn:[2,13],cahn_hillard_form:13,calcul:[],call:[2,5,6,13,14,18,19,21],came:5,can:[1,2,3,5,6,7,8,9,13,14,15,17,18,19],cancer:[0,1,2,4,8,22,23,24,25],cannot:2,canon:5,capillari:[2,6,13,14,24,25],capilllari:2,care:[2,3,5,7,11],carlo:0,castro:0,caus:2,cdot:[2,5,6,13,14,25],cell:[2,3,6,11,14],cell_radiu:11,center:[2,5,6,7,11,14,15,25],central:25,certain:2,cg1_element:6,cg:[2,3,6,7,17],cgi:0,chang:[2,3,6,7,8,9],chapter:[],check:[2,3,5,11,14,17,20],checker:11,chem_potenti:13,chem_potential_const:23,chemic:[5,23],chempot_const:[6,7,23],chi:[2,6,7,14,23],choic:[3,5,7,17],choos:[5,6,7],chose:6,chosen:6,chri:0,chunk:17,circl:[2,3,5,11,14,15],circle_radiu:11,circular:15,cite:[2,5,23,24],clarif:2,clarifi:6,clean:9,clear:9,clearli:5,click:[2,3,4,5,6,7],clock:11,clock_check:11,clockcheck:11,close:5,closer:2,closest_tip_cel:14,cma:0,coarsen:0,code:[4,8,14,19],coher:14,collabor:[3,6,7,10,23,24],collaps:[2,3,6,7],collect:5,column:20,com:9,come:2,comm:[2,3,6,7,17,21],comm_world:[2,3,6,7,17],command:[2,6,9,19],comment:6,common:[6,9,13,23],commun:[9,17],compar:3,compil:9,complet:[1,2,5,6,8,23,24],complex:2,compon:6,compos:[2,6,17],compromis:[],compulsori:20,comput:[0,3,5,6,7,9,14,17],compute_phi_c:14,compute_tip_cell_veloc:14,conacyt:0,concentr:[2,5,6,13,14],condit:[11,13,14,24],conductor:5,confgure_root_logger_with_standard_set:18,confid:[],configur:[6,7,18],conserv:17,consid:[2,3,5,6,7,9,13],constant:[2,3,5,6,13,14,23,24],constant_valu:[],constantafexpressionfunct:[],constantli:[2,6],constantsourcesfield:11,constrain:[3,7],construct:[5,6],constructor:2,consum:[6,24],consumpt:[2,13],contain:[2,6,8,10,11,14,16,17,19,20,22,23,24],content:1,contextu:18,continu:[2,6],continuo:17,contributor:0,conveneinc:13,conveni:[2,6],coordin:2,copyright:9,core:[6,20],correctli:9,correspond:[13,20,23,24],correspong:23,corvera:0,costant:13,could:[2,13],count:2,coupl:[0,6],cours:[3,5,6,7,9],cpp:[11,13,14,17,21,23],cpu:[2,3,6,7],creat:[2,3,5,6,7,9,11,14,19,20],creation_step:[11,14],credit:9,critic:5,csv:18,cube:5,current:[1,2,5,6,7,9,11,14,17,19],current_step:[6,7,14],cylinder_radiu:3,cylindr:3,d:[0,2,3,4,5,6,9,13],d_:[2,24],d_sigma:24,data:[1,2,3,5,7,9,16,19],data_fold:[2,3,6,7,17,18,19],datafram:20,date:19,dateandtim:19,dc:13,de:0,deactiv:[2,14],debug:18,debugadapt:18,decai:[6,23],decemb:0,decid:[5,19],decreas:2,defin:[1,2,3,5,6,7,13,14,17,18,21,23,25],definit:13,defint:2,degre:[2,3,5,6,7,17],deleg:[],deliber:5,delta:[6,7,23],demo:[2,3,5,6,7,8,13],demo_doc:[],demo_doc_jupyt:4,demo_doc_python:4,demo_in:[2,3],demo_out:[2,3,5,6,7],depend:[2,5,6,9],deprec:17,deriv:[2,5,6,13,23,25],desc:[2,3],describ:[2,3,5,6,7,11,23,24],descript:[23,24],desid:17,design:3,desktop:[3,7],detail:[5,13],determin:15,develop:[5,6,8],df:[6,13],df_dphi:23,dict:[19,20,21,24],dictionari:[6,20],did:[2,3,6,7],differ:[2,3,5,6,7,9,14,16,17],differenti:[0,3,5,6,7],difficult:[],diffus:[0,2,6,13,23,24],dimens:[3,6,7],dimenson:[],directli:[9,21],directori:[9,19],discours:[],discret:[11,13,23,24],discuss:[2,5,6,13],distanc:[2,14],distant:2,distibut:[],distribut:[2,5,6,11,12,13,17],divid:[1,2,17],divide_in_chunk:17,divis:17,docker:9,document:[5,6,9,23],doe:9,doesn:9,doi:0,dokken:5,dolfin:[11,13,14,17,21,23,24,25],domain:[0,2,5,6,7,11,14],don:[2,3,7,9],done:[6,7],dot:5,doubl:[6,7,23,24],down:6,download:[2,3,4,5,6,7],driven:[2,6],drop:2,dt:[2,3,5,6,7,13,23,24],duplic:6,dure:[2,3,6,7,19],dx:5,dynam:2,e0149422:0,e19989:0,e:[0,2,5,6,11,14,25],each:[1,2,5,6,11,13,14,17,19],eas:[3,7],easi:[2,6],easier:[2,9],easiest:6,easili:[2,6],easli:[2,3,6,7],east:11,effect:2,effici:[3,5,6,8],effort:[3,7],egg:9,either:18,elabor:17,element:[2,5,6,7,13,17],element_typ:17,elimin:[],ellips:[7,15],ellipsefield:[6,15],ellipsoid:[7,15],ellipsoidfield:[7,15],ellipt:6,elment:5,els:[2,3,6,7,14],empti:6,en:0,encourag:5,encyclopedia:0,end:[],endotheli:2,energet:13,engin:0,enough:9,ensur:5,enter:9,entir:[2,5,6],epsilon:[2,6,7,13,23],equal:[2,11,17,19,23],equal_for_all_p:[],equat:[0,3,6,7,13,23,24],error:[2,3,6,19],error_msg:19,es:0,especi:19,espress:[],estim:[6,25],estimate_cancer_area:25,estimate_capillaries_area:25,et:[2,3,6,11,13,14,24],etc:[2,9,16],eugenia:0,euler:[5,23,24],ev:12,eval:14,evalu:14,even:[3,6,7,17],everi:[5,6,14],everyth:[2,3,6,9,20],everywher:2,evolut:[5,6,24],exactli:[2,3,6,7],exampl:[4,5,17],excel:[5,6],except:[2,3],execut:[9,19],execution_tim:19,exist:[2,3],expect:2,experi:6,experienc:[5,6],explain:[2,5],exploit:[2,3,6,7],express:[1,2,3,5,6,7,8,11,14],extens:[2,5,6],extra:18,extrem:[2,3,5],ey:[],f:[2,3,6,7,13,21],fact:[5,9],factor:[2,11,13,14],fals:[2,3,6,7,11,14,17,20],familiar:[],far:2,faster:[],fct:0,featur:[5,6],feb:0,feel:5,fem:[2,5,6],fenic:[0,1,2,3,4,6,7,8,10,11,13,14,17,23,24],fenicsproject:9,fenicsx:[5,9],fenut:[1,2,3,6,7,8,11,13,14,23,24],few:[5,6],ffor:18,fibonacci:12,fibonacci_spher:12,fick:[0,2,6],field:[0,1,2,3,4,6,7,8,10,11,13,14,15,19,22,23,24,25],figur:5,file:[2,3,5,6,7,9,16,17,18,19,20],file_af:[2,3],file_c:[2,3],file_fold:[2,3,6,7],file_nam:[2,3,17],find:[1,2,3,5,6,7,8,13],finit:[2,5,7,13],finiteel:6,fir:[],first:[2,3,5,6,9,19],flat:17,flatten:17,flatten_list_of_list:17,flow:5,fluid:5,fo:23,folder:[2,3,6,7,9,17,18,19],folder_nam:19,folder_path:[2,3,6,7,19],follow:[1,2,3,5,6,7,8,9,13,14,19],follw:13,form:[0,1,3,7,10,20,23,24],form_af:[2,3],form_ang:[2,3],format:6,formul:0,formula:[5,14],forum:6,found:14,fpradelli94:9,frac:[2,5,6,13,14,25],frame:20,fredrik:[],free:[0,5],frequenc:0,from:[2,3,5,6,7,9,11,18,20,24],from_dict:[6,7,20],from_ods_sheet:[2,3,20],fu:[2,3],full:[1,9,11],fun:[2,3],function_spac:[2,3,6,7],functionspac:[5,6,25],further:13,futur:9,g:[0,2,5,6,7,14],g_m:[2,14],galerk:6,galleri:[2,3,5,6,7,8],gamg:[3,6,7],gamma:[5,6,7,23],garth:0,ge:2,gear:5,gener:[2,3,4,5,6,7,11,13,17,19,24],genericmatrix:21,genericvector:21,get:[2,3,5,9,11,14,20],get_dim:[],get_global_mesh:[],get_global_source_cel:11,get_global_tip_cells_list:14,get_latest_tip_cell_funct:[2,3],get_local_mesh:[],get_local_source_cel:11,get_mixed_function_spac:[2,3,6,7,17],get_point_value_at_source_cel:[],get_radiu:14,get_rank:[2,3,6,7],get_valu:[2,3,6,7,20],gif:[],git:9,github:9,give:5,given:[2,3,5,6,7,11,14,15,17,19,20,25],given_list:17,given_valu:2,glaerkin:17,global:[11,14],gmre:[3,6,7],go:[5,6,9],goe:6,gomez:0,good:[5,9],grad:[2,3,5],grad_af:[2,3,14],grad_af_function_spac:[2,3],grad_t:[2,3],gradient:[2,13,14],gradual:5,gram:[6,7],great:[3,7],greatest:5,group:[],grow:2,growth:[0,2,6],guarante:7,guillermo:0,gulbenkian:0,h:[0,13,24],ha:[3,5,9,11,14,15],hake:0,han:0,hand:2,handl:[2,13],happen:2,hard:6,have:[2,3,5,6,7,9,11,19,20,25],healthi:24,heavisid:13,heavysid:25,hector:0,heidelberg:0,help:9,here:[2,3,5,6,7,9],hern:0,hi:3,hierarch:0,high:[2,5],higher:[2,3,7],hillard:[2,13],home:9,host:[2,9],how:[4,5],howev:[2,3,5,6,7,9,13],hpc:[3,7],html:19,http:[0,9],hugh:0,hurri:[],hybrid:10,hypothesi:6,hypox:[2,24],i:[2,6,9,11,14],i_v_w:2,id:0,ident:[3,6,7,9],identifi:[17,19],ignor:9,imag:[],implement:[5,8,11,13,14],implment:4,imput:[],includ:[2,6,8],inde:[2,3,5,6,7,13,19],index:[0,8],induc:[2,11],inf:25,info:[16,18,19],infocsvadapt:18,inform:[2,6,9,18,20],inhibit:2,init:[3,6,7],initi:[13,23,24],initial_vessel_radiu:3,initial_vessel_width:[2,3],input:[2,5,19],input_mesh:[],insert:18,insid:[2,3,5,6,9,14,15,17,20],inside_valu:[6,7,15],inspect:[6,8],instal:[2,3,6,7,8],instanc:[2,19,20],instead:[3,6,7,17],instruct:9,int_:5,integr:8,interact:2,interest:[2,5,6,17],interfac:[0,2,3,5,6,24],intern:[14,18,21],interpol:[2,3,6,7],intracomm:[17,21],introduc:2,introduct:4,introduction_to_fen:5,invit:5,involv:[2,13],io:9,ipynb:[2,3,5,6,7],ipython:9,is_in_ellipse_cpp_cod:[],is_in_local_box:17,is_inside_global_mesh:[],is_inside_local_mesh:[],is_paramet:20,is_point_insid:14,is_point_inside_mesh:17,is_value_pres:20,isbn:0,isciii:0,iter:[2,3,5,6,7],its:[2,6,9,13,14,18,23],itself:[2,6,13,18],j:[0,2,3,5,6,7,21],j_mat:[3,6,7],jacobian:[2,6],jan:0,jessica:0,jiangp:0,job:[2,6],johan:0,johann:0,johansson:0,journal:0,json:17,juan:0,jun:0,jupyt:[2,3,4,5,6,7],just:[2,3,4,5,6,7,9,11,19],k:0,keep:[2,11],kehlet:0,kei:[2,6],kevin:0,keyword:18,kind:[5,13],know:5,known:6,kristian:0,krylov:[],ksp_type:[3,6,7],kwarg:[11,18],l:5,la:21,lagrang:[5,13],lambda:[2,3,6,7,11,13,23],lambda_:24,lambda_phi:24,langtangen:0,langtangenlogg2017:5,languag:[0,1,5],laptop:[3,7],larg:6,larson:[],last:[2,5],later:9,latest:9,latter:[2,13],law:0,lb13:[],le:[2,13,14],lead:[2,5,13],least:20,leav:6,left:[2,6],legaci:[1,8,22],lei:0,length:[],let:[2,5,6],level:[2,3,5,18],leverag:[5,6],libexec:9,librari:6,licens:9,like:[4,5,6,9],limit:2,line:[2,5,6,19],linear:[2,6,21],linearli:2,link:[2,3,6,7],linux:8,list:[6,9,11,14,17,23],list_of_list:17,liter:[6,7],litform:[1,6,7,8],liu:0,ll17:[0,5],ll:[3,7,18],lmbda:13,load:[3,20],load_paramet:17,load_random_st:[],local:[2,9,11,17],local_box:17,local_mesh:17,locat:7,log:[1,2,3,6,16],logg:0,logger:[5,18],loggeradapt:18,loglevel:[2,3,6],longer:3,look:[5,6],loop:5,lorenzo:[0,6,7,23],low:[2,6],lower:[7,13],lst:[0,6,7,23],lx:[2,3],ly:[2,3],lz:3,m:[0,2,6,7,9,13],m_:24,m_c:2,m_phi:24,machado:0,made:3,magnitud:[3,7],mai:[9,17],main:[2,6,9,13],mainli:[5,18],make:[2,3,5,6,7,13],manag:[1,2,3,11,14,16,19],mani:[],manipul:18,mansimd:[2,3],mansimdata:[1,2,3,6,7,16],manzanequ:0,map:[2,3,11],mar:0,mari:0,mario:0,martin:0,mat:[3,6,7],matemat:[],math:[0,1,2,3,6,7,8,13],mathemat:[0,5,8],matrix:6,matter:6,max:[6,25],maximum:13,mcte:0,mean:[2,6],measur:[2,20],mechan:0,ment:5,mention:[2,5,6],merg:2,mesh:[3,11,14,17],mesh_dim:14,mesh_fil:3,mesh_wrapp:[],mesh_xdmf:3,meshwrapp:[],messag:[6,9,18,19],met:[11,14],method:[0,1,2,5,6,9,11,13,14,16,17,18,19,23,24],metod:5,michael:0,micinn:0,might:[3,5,6,7,9,17],migth:[],min:6,min_tipcell_dist:14,minut:[2,3,5,6,7],mistak:2,mix:[6,9,17],mixed_el:6,mixedel:6,mkdir:[2,3,9],moacf:9,mobil:24,mocaf:[1,4,5],mocafe_fold:[],mocef:[],model:[0,1,3,4,5,7,8,10,11,13,14,22,23,24],modifi:[2,18],modul:[6,8,10,11,14,17,18,22,23,24],moment:[],monitor:6,more:[2,5,6,9,11,14,17],moreov:2,most:[3,5,6,7],move:[0,2,3,14],move_tip_cel:[2,3,14],mpar:[2,3],mpi4pi:[17,21],mpi:[2,3,6,7,8,9,11,14,17,19,21],mpi_comm:[],mpirun:[2,3,6,7,9],msg:18,mshr:2,mu0:13,mu:[2,3,6,7,13],mu_0:[2,3],much:[],multidimension:[],multipl:[11,19],must:[6,20,23,24,25],mv:[2,3],mx:0,my:[],my_sim_data:19,n:[0,2,3,5,6,7],n_chnk:17,n_global_mesh_vertic:[],n_local_mesh_vertic:[],n_point:12,n_sourc:[2,3,11],n_step:[2,3,6,7],n_variabl:17,nabla:[2,5,6,13,14],name:[2,6,17,19,20,23,24],name_param_a:20,name_param_b:20,nation:0,natur:[2,3,5,6,7],ncol:[2,3,6,7],ndarrai:[11,15],ndez:0,necessari:[2,6,9],need:[2,3,5,6,7,9,17,18,19],network:2,neumann:[2,3,5,6,7],never:5,new_posit:14,new_valu:20,newton:[],newtonl:[3,6,7],newtonsolv:21,next:[2,5,6,7],nl:21,non:[6,11],none:[2,3,6,7,11,19,21],nonlinear:[6,21],nonlinearproblem:21,norm:[2,14],normal:[3,7,17,18],notch:2,note:9,notebook:[2,3,4,5,6,7],noth:[5,6,11,14,18,19],notic:[2,3,5,6,7,9,17],now:[2,3,6,9],np:[6,7],number:[2,3,5,6,7,12,17],numer:[0,6],numpi:[6,7,9,11,15],nutrient:[2,6,23,24],nutrient_weak_form:[],nx:[2,3,6,7],ny:[2,3,6,7],nz:[3,7],object:[2,3,5,6,11,17,19,20,21,23,24],obscur:6,obtain:5,occur:[2,6,9,19],occurr:[],od:[2,3,9],odf:20,off:[2,3],offici:9,often:[2,5],old:5,oldid:0,omega:[2,5,13],onc:[2,3,6,7],one:[2,3,5,6,7,13,14,17,18,19],onli:[2,3,6,9,12,14,18],onlin:[0,5],onward:9,open:[5,8,9],openmpi:9,oper:[2,5,6,9],optim:6,option:[2,3,6,7,9,11,19],order:[2,3,5,6,7,13,17,19,25],ore:[23,24],org:[0,9],origin:[2,5,6,13,23,24],other:[2,5,6,9,11,14,20],otherwis:[2,11,14,17,19,20],our:[2,5,6,9],out:[6,9],outlin:6,output:9,outsid:[5,6,15],outside_valu:[6,7,15],over:17,overrid:18,own:[6,18],oxygen:2,p0:[],p:[0,13,14,24],packag:[1,5,6,8,9],page:[2,3,6,7,8,9],panda:[9,20],paper:[2,6,13,23,24],parallel:[2,3,6,7,8,17,19],param:[17,24],param_df:20,param_dict:20,paramet:[1,2,3,6,7,8,11,12,13,14,16,17,18,19,23,24,25],parameters_fil:[2,3,17],paraview:[2,3,5,6,7],parent:[2,3,6,7],part:[2,11],partial:[0,2,5,6,13],particular:0,pass:[2,18],path:[2,3,6,7,18,19,20],pathlib:[2,3,6,7,18,19,20],pathwai:2,pattern:[0,6],pbar:[2,3],pc:9,pc_model:[6,7],pc_type:[3,6,7],pde:[0,2,5,6],peopl:5,perfectli:5,perform:[2,6],person:0,perspect:[],petsc4pi:[3,6,7],petsc:[3,6,7,21],petscmatrix:[3,6,7],petscnewtonsolv:21,petscproblem:[6,21],petscvector:[3,6,7],petter:0,phase:[0,1,2,3,4,6,7,8,10,11,13,14,15,22,23,24,25],phi0:[5,6,7],phi0_cpp_cod:[],phi0_in:[6,7],phi0_max:[],phi0_min:[],phi0_out:[6,7],phi:[5,6,7,23,24,25],phi_0:24,phi_c:14,phi_prec:23,phi_th:14,phi_xdmf:[5,6,7],philosophi:3,php:0,physic:5,pi:[2,14],pick:[2,6],pictur:6,piec:5,pip3:9,pip:9,place:[2,3,5,6,9,11,17,18],plan:9,platform:[4,5],pleas:[6,23,24],plo:0,ploson:0,pna:[0,6],png:[],point:[0,2,3,5,6,7,11,12,14,17],poir:0,polynomi:3,pone:0,popul:2,popular:5,portabl:6,posit:[2,3,11,14,17],possibl:[3,5,7,13,14],post:6,potenti:[13,23,24],pow:[3,5],power:[3,5,7],ppa:9,practic:[2,5],precis:[6,11,14,17],precondition:[],preconditioner_typ:3,prefer:[17,21],presenc:[5,6],present:[2,6,7,10,11,14,20,23],pretti:[6,24],previou:2,print:6,problem:[0,3,6,7,9,21],proce:[2,3,9],procedur:[9,14],proceed:0,process:[2,3,6,7,11,14,17,18],product:24,progress:[2,3,6,7],progress_bar:[6,7],project:[0,2,3,5],prolif:24,prolifer:[2,6,13,14,23],properli:9,properti:9,prostat:[0,2,4,23],prostate_canc:[1,6,7,22],prostate_cancer2d:6,prostate_cancer3d:7,prostate_cancer_2d:6,prostate_cancer_3d:7,prostate_cancer_chem_potenti:23,prostate_cancer_form:[6,7,23],prostate_cancer_nutrient_form:[6,7,23],prostate_cancer_weak_form:[],provid:[2,3,4,5,6,8,9,11,14,17,19],pseudo:14,pt:0,publish:6,purpos:[2,6,8,9,14,17,18],put:[5,20],pvd:17,py:[2,3,5,6,7],pyhton:[],python3:[2,3,6,7,9],python:[0,2,3,4,5,6,7,8,9,15,18,20],python_fun:15,python_fun_param:15,pythonfunctionfield:15,q:13,quad:[2,5,13,14],quai:9,quit:2,r:[0,5,9,14,25],r_c:[2,14],r_v:3,radiu:[2,11,12,14,15],rand_max:[6,7],random:[6,7,11],random_sources_domain:2,randomli:[2,11,14],randomsourcemap:[2,3,11],randomst:[],rang:[2,3,5,6,7],rank:[2,3,6,7],rate:[2,6,13,14,23,24],rational:19,re:[2,3,5,6,9],reach:[],read:[2,3,5,6,13,14,25],reader:9,readi:[6,9],real:6,realli:[3,6,7],reason:[2,3,7,9],recommend:[2,3,5,6,7,8],rectangl:[2,5],rectanglemesh:[2,6],reduc:[2,6,7],refer:[2,6,20,23,24],refin:0,regard:[5,6,9],rel:[2,17],relat:[1,6,13],reload:3,remain:[2,3,7,13],remaind:17,remark:7,rememb:[2,3,7,23,24],remov:[2,11,14],remove_global_sourc:11,remove_sources_near_vessel:[2,3,11],report:[2,6,9,13,14,20,23],repositori:9,repres:[2,5,6,11,13,14,15,17,20,24,25],represent:5,reproduc:[2,6,8,11],requir:[2,3,5,6,7,9,17,23,24],resembl:[3,7],resolv:[2,3,6,7],respect:[2,6,11],respons:[2,11],result:[5,9,11,17,19],retriev:[2,6],revert:[2,3],revert_tip_cel:[2,3,14],revis:2,rf:9,richardson:0,right:[2,14],rim:24,ring:0,risk:[2,9],rm:9,rmw:[],rodrguez:0,rogn:0,root:[2,18,19],rotational_expression_function_paramet:[],rotationalafexpressionfunct:[],routin:6,row:5,rui:0,rule:2,run:5,s:[0,2,3,5,6,7,23],s_:6,s_av:[6,7],s_averag:[6,7],s_exp:[6,7],s_express:[],s_max:[6,7],s_min:[6,7],salt:5,same:[2,3,6,7,10,13,17,19],save:[2,3,5,6,7,19],save_random_st:[],save_sim_info:19,scalar:13,scale:[0,6],scienc:0,scientif:[5,6,23,24],scott:0,script:[2,3,4,5,6,7,8,9],search:8,second:[2,3,5,6,7,13,19],section:[2,5,9],secur:9,see:[2,3,4,5,6,7,9,13],seen:2,select:[2,14],self:[2,6,21],semi:5,semiax:[6,15],semiax_i:[6,7,15],semiax_x:[6,7,15],semiax_z:[7,15],separ:[2,3,9,13,24],set:[2,3,5,6,7,15,18,19,20],set_descript:3,set_equal_randomstate_for_all_p:[],set_log_level:[2,3,6],set_valu:20,setfromopt:[3,6,7],setfunct:[3,6,7],setjacobian:[3,6,7],setstat:[],setup_data_fold:[2,3,6,7,19],setup_pbar:3,setup_pvd_fil:17,setup_xdmf_fil:[2,3,6,7,17],sever:6,share:[],sheet:[2,20],shell:9,shf:25,should:[5,9],show:[2,3,5,6,7],shown:2,shut:6,side:[2,3,5,6,7],sif:9,sigma0:[6,7],sigma0_in:[6,7],sigma0_out:[6,7],sigma:[6,7,23,24],sigma_0:24,sigma_old:23,sigma_xdmf:[6,7],sigmoid:[15,25],signal:2,significantli:6,sim_data:19,sim_info:19,sim_nam:19,sim_rational:19,sim_valu:20,similar:3,similarli:[2,6],simparam:[2,3],simpi:9,simpl:[2,5,6,25],simplest:5,simpli:[2,3,5,6,7,9,13,19,24],simplier:[],simul:[0,1,5,8,10,11,13,14,16,17,19,20,24],sinc:[2,5,6,9,14],singl:[2,5],singular:8,singularityc:9,situat:5,slightli:[7,14],slope:[15,25],small:3,smooth:[15,25],smoothcircl:15,smoothcirculartumor:15,sne:[3,6,7],snes_solv:[3,6,7],snes_typ:[3,6,7],snesproblem:[3,6,7],so:[1,2,3,5,6,9,13,19],softw:0,softwar:[0,3,5,7,9],solut:[2,3,6,7,9],solv:[0,2,3,6,7,13],solvabl:[2,13],solver:[1,3,6,7,16],solver_paramet:21,solver_setup:21,solver_typ:3,some:[4,5,6,9,13,17],someth:[6,9],sometim:5,soon:2,sourc:[2,3,4,5,6,7,8,9,11],source_cel:11,source_map:11,source_point:11,sourcecel:11,sourcecellsmanag:2,sourcecomput:[],sourcemap:[2,11],sources_in_circle_point:11,sources_manag:[2,3],sources_map:[2,3],sourcesfield:[],sourcesmanag:[2,3,11],space:[2,5,6,7,17,25],spatial:[11,14],speak:[5,6],specif:[0,3,4,5,6,7,13,18],specifi:[5,6,19,23,24],sphere:[3,12,14],sphinx:[2,3,4,5,6,7],sphinx_gallery_thumbnail_path:[],spiral:[],spline:0,split:[2,3,6,7,13],spread:12,springer:0,sprout:2,squar:[2,5,6,17],stabl:9,standard:[2,3,6,13,18],standerd:18,start:[0,2,3,5,6],start_point:11,starv:2,state:[],statement:[2,3],std_out_all_process:[2,3],stem:2,step:[2,3,4,5,6,7,13,14,23,24],still:6,stop:[2,9],store:[2,5,7,17,19,20],str:[17,18,19,20],string:[5,17],strong:0,structur:[5,20],studi:[],stuff:6,sub:[2,3,6,7],subclass:18,submodul:1,subpackag:[1,10,16,22],subspac:2,subsystem:9,sudo:9,suffici:2,suggest:[2,6],suit:[6,18],sum:[2,13],summar:[1,5],summari:5,suppli:[6,23],support:9,suppos:6,sure:[2,3,6,7],surpass:11,surround:11,surviv:2,sy:6,symbol:5,sympli:[],system:[2,5,6,8],t:[0,2,3,5,6,7,9,13,24],t_c:14,t_valu:14,take:[2,3,6,7,9,11],tast:[],tau:[6,7,23],tc_p:14,team:13,techniqu:5,tell:[6,9],term:[2,5,6,13],test:[2,3,9,11,13,23,24],testfunct:[2,3,5,6,7,13,23,24],tew:0,textrm:[2,5,13,14],than:[2,3,5,6,7,14,17],thei:[2,3,6,7,11],them:[1,2,3,6,7,8,9],theori:[],theta:[2,13],thi:[4,5,9,10,11,13,14,16,17,18,19,21,22,23,24],thing:[2,3,5,6,7,9],think:[],third:[2,5],thoma:0,thorugh:2,those:[5,6,9],though:[3,6,17],three:2,threshold:[2,11],through:2,throughout:[2,6,14],thu:[2,3,7,9,14],time:[2,3,5,6,7,9,11,13,19,23,24],tip:[2,3,14],tip_cel:14,tip_cell_manag:[2,3],tip_cell_posit:14,tipcel:[1,2,3,10],tipcellmanag:[2,3,14],tipcells_field:[2,3],tipcells_xdmf:[2,3],tipcellsfield:14,tipcellsmanag:[2,3],tissu:[0,6,11,13,24],titl:0,todo:[],togeth:[2,5],told:9,too:[],took:2,tool:[11,14],toolkit:6,topic:5,total:[2,3,5,6,7,17],toward:6,tpoirec:[0,2,3,10,11,13,14],tqdm:[2,3,6,7,9],tradit:[],tran:0,transform:[],transit:9,translat:[5,6,11],travasso2011:[],travasso2011a:2,travasso:[0,2,3,10,11,13,14],trial:[],trialfunct:5,triangl:6,tumor:[0,6,15,24],turn:[2,3],tutori:5,two:[2,3,6,13,14],type:[5,6,9,17,19],u:[2,3,5,6,7,24],u_0:5,u_0_exp:5,u_xdmf:5,ub:0,ufl:[2,5,6,13,23,24],um:[6,7],under:19,understand:[5,8],uni:0,unifi:[0,1,5],uniform:6,uninstal:8,uniqu:2,unit:[2,20],unitsquaremesh:5,unknown:13,up:[2,3,5,6,7,9],updat:[2,3,5,6,7,9,14],uptak:[23,24],url:0,us:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,17,18,19,20,21,23,24,25],usabl:5,user:[2,5,6,9,15,18,19],usernam:9,usr:9,usual:[2,17],util:[1,16],v1:[2,3,6,7,13],v2:[2,3,6,7,13],v3:[2,3],v:[2,5,13,14,23,24,25],v_:24,v_pc:24,v_uh:24,v_ut:24,valu:[2,5,6,7,13,14,15,20,23,24,25],value_param_a:20,value_param_b:20,var_phi:23,variabl:[2,3,5,6,13,17,23,24,25],variant:5,variat:[3,6],varibl:25,varphi:[6,7,23,24,25],vascular:[0,2,11,13],vascular_proliferation_form:13,ve:7,vec:[3,6,7],vector:[2,3,6,7,14],vectorfunctionspac:[2,3],vegf:2,veloc:[2,14],velocity_norm:14,veri:[2,5,6],versatil:24,version:[0,3,5,7,9],vessel:[2,3,11,25],view:0,vilanova:0,visibl:[],visual:[5,17],vol:0,w:0,wa:[2,6,23],wai:[3,5,6,7,9,13,17,21],wander:9,wank:5,want:[2,5,6,9,17],warn:9,water:5,we:[2,3,5,6,7,9],weak:[0,1,3,7,13,23,24],weak_form:[2,3,6,7],websit:5,weight:13,well:[0,6,7,9,18,23,24],what:[2,6,7,9],when:[2,11,14,17,19],where:[2,3,4,5,6,11,13,14,17,18,25],which:[2,5,6,9,11,13,14,15,17,20,24,25],who:5,why:9,width:[2,17],wikipedia:0,wikipediacontributors21:[0,2,6],window:9,wise:5,wish:6,without:[2,6,19],won:9,wonder:[],word:5,work:[3,5,6,9,17,19,23,24,25],workflow:6,wrap:2,wrapper:[11,20],write:[2,3,5,6,7],written:5,wrong:6,wsl:9,www:[0,9],x:[2,3,5,6,14,25],x_max:[6,7],x_min:[6,7],xdmf:[2,3,5,6,7,17],xdmffile:5,xu16:[1,22],xu2016_nutrient_form:24,xu:[0,24],xu_2016_cancer_form:24,xvg16:[0,24],y:[9,25],y_max:[6,7],y_min:[6,7],year:[0,6,7],yongji:0,you:[1,2,3,4,5,6,7,8,9,17,18,19,23,24],your:[2,3,5,6,7,8,13,18,23,24],yourself:[5,6],z_max:7,z_min:7,zero:[2,6],zhang:0,zip:4},titles:["Bibliography","Code Documentation","Angiogenesis","Angiogenesis 3D","Demo Gallery","A brief introduction to FEniCS","Prostate cancer","Prostate cancer 3D","Welcome to <em>mocafe</em>\u2019s documentation!","Installation","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.angie</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.angie.af_sourcing</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.angie.base_classes</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.angie.forms</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.angie.tipcells</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.expressions</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.fenut</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.fenut.fenut</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.fenut.log</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.fenut.mansimdata</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.fenut.parameters</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.fenut.solvers</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.litforms</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.litforms.prostate_cancer</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.litforms.xu16</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.math</span></code>"],titleterms:{"3d":[3,7],"function":3,A:5,In:[],The:[5,10,16,22],af_sourc:11,agent:2,angi:[10,11,12,13,14],angiogenesi:[2,3],apt:9,base:2,base_class:12,basic:5,below:[10,16,22],bibliographi:0,boundari:[2,3,5,6,7],brief:[2,5,6],can:[],cancer:[6,7],code:[1,2,3,5,6,7],complet:[],comput:2,condit:[2,3,5,6,7],contain:9,content:[],definit:[2,3,5,6,7],demo:4,differenti:2,diffus:5,discret:[2,5,6,7],document:[1,8,10,16,22],domain:3,each:[10,16,22],equat:[2,5],exampl:[2,3,6,7],express:15,fenic:[5,9],fenut:[16,17,18,19,20,21],field:[],find:[],follow:[],form:[2,5,6,13],full:[2,3,5,6,7,10,16,22],galleri:4,heat:5,how:[2,3,6,7],implement:[2,3,6,7],includ:[],index:[],indic:8,initi:[2,3,5,6,7],instal:9,introduct:[2,5,6],linux:9,litform:[22,23,24],log:18,mansimdata:19,math:25,mathemat:[2,6],mesh:[2,5,6,7],mocaf:[2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],model:[2,6],note:[3,7],packag:[],paramet:20,pde:[3,7],phase:[],problem:5,prostat:[6,7],prostate_canc:23,provid:[10,16,22],recommend:9,remov:9,result:[2,3,6,7],run:[2,3,6,7],s:8,setup:[2,3,5,6,7],simul:[2,3,6,7],singular:9,solut:5,solv:5,solver:21,space:3,spatial:[2,3,5,6,7],submodul:[10,16,22],system:[3,7,9],tabl:8,thi:[2,3,6,7],tipcel:14,uninstal:9,visual:[2,3,6,7],weak:[2,5,6],welcom:8,what:5,workflow:5,xu16:24,you:[],your:9}})