Search.setIndex({docnames:["bib","code_documentation","demo_doc/angiogenesis_2d","demo_doc/angiogenesis_3d","demo_doc/index","demo_doc/introduction_to_fenics","demo_doc/multiple_angiogenesis_simulations","demo_doc/multiple_pc_simulations","demo_doc/prostate_cancer2d","demo_doc/prostate_cancer3d","index","installation","sub_code_doc/angie","sub_code_doc/angie_sub_code_doc/af_sourcing","sub_code_doc/angie_sub_code_doc/base_classes","sub_code_doc/angie_sub_code_doc/forms","sub_code_doc/angie_sub_code_doc/tipcells","sub_code_doc/expressions","sub_code_doc/fenut","sub_code_doc/fenut_sub_code_doc/fenut","sub_code_doc/fenut_sub_code_doc/log","sub_code_doc/fenut_sub_code_doc/mansimdata","sub_code_doc/fenut_sub_code_doc/parameters","sub_code_doc/fenut_sub_code_doc/solvers","sub_code_doc/litforms","sub_code_doc/litforms_sub_code_doc/prostate_cancer","sub_code_doc/litforms_sub_code_doc/xu16","sub_code_doc/math"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinxcontrib.bibtex":9,sphinx:56},filenames:["bib.rst","code_documentation.rst","demo_doc/angiogenesis_2d.rst","demo_doc/angiogenesis_3d.rst","demo_doc/index.rst","demo_doc/introduction_to_fenics.rst","demo_doc/multiple_angiogenesis_simulations.rst","demo_doc/multiple_pc_simulations.rst","demo_doc/prostate_cancer2d.rst","demo_doc/prostate_cancer3d.rst","index.rst","installation.rst","sub_code_doc/angie.rst","sub_code_doc/angie_sub_code_doc/af_sourcing.rst","sub_code_doc/angie_sub_code_doc/base_classes.rst","sub_code_doc/angie_sub_code_doc/forms.rst","sub_code_doc/angie_sub_code_doc/tipcells.rst","sub_code_doc/expressions.rst","sub_code_doc/fenut.rst","sub_code_doc/fenut_sub_code_doc/fenut.rst","sub_code_doc/fenut_sub_code_doc/log.rst","sub_code_doc/fenut_sub_code_doc/mansimdata.rst","sub_code_doc/fenut_sub_code_doc/parameters.rst","sub_code_doc/fenut_sub_code_doc/solvers.rst","sub_code_doc/litforms.rst","sub_code_doc/litforms_sub_code_doc/prostate_cancer.rst","sub_code_doc/litforms_sub_code_doc/xu16.rst","sub_code_doc/math.rst"],objects:{"mocafe.angie":[[13,0,0,"-","af_sourcing"],[14,0,0,"-","base_classes"],[15,0,0,"-","forms"],[16,0,0,"-","tipcells"]],"mocafe.angie.af_sourcing":[[13,1,1,"","ClockChecker"],[13,1,1,"","ConstantSourcesField"],[13,1,1,"","RandomSourceMap"],[13,1,1,"","SourceCell"],[13,1,1,"","SourceMap"],[13,1,1,"","SourcesManager"],[13,3,1,"","sources_in_circle_points"]],"mocafe.angie.af_sourcing.ClockChecker":[[13,2,1,"","clock_check"]],"mocafe.angie.af_sourcing.SourceMap":[[13,2,1,"","get_global_source_cells"],[13,2,1,"","get_local_source_cells"],[13,2,1,"","remove_global_source"]],"mocafe.angie.af_sourcing.SourcesManager":[[13,2,1,"","apply_sources"],[13,2,1,"","remove_sources_near_vessels"]],"mocafe.angie.base_classes":[[14,1,1,"","BaseCell"],[14,3,1,"","fibonacci_sphere"]],"mocafe.angie.base_classes.BaseCell":[[14,2,1,"","get_dimension"],[14,2,1,"","get_distance"],[14,2,1,"","get_position"]],"mocafe.angie.forms":[[15,3,1,"","angiogenesis_form"],[15,3,1,"","angiogenic_factor_form"],[15,3,1,"","cahn_hillard_form"],[15,3,1,"","vascular_proliferation_form"]],"mocafe.angie.tipcells":[[16,1,1,"","TipCell"],[16,1,1,"","TipCellManager"],[16,1,1,"","TipCellsField"]],"mocafe.angie.tipcells.TipCell":[[16,2,1,"","get_radius"],[16,2,1,"","is_point_inside"],[16,2,1,"","move"]],"mocafe.angie.tipcells.TipCellManager":[[16,2,1,"","activate_tip_cell"],[16,2,1,"","compute_tip_cell_velocity"],[16,2,1,"","get_global_tip_cells_list"],[16,2,1,"","move_tip_cells"],[16,2,1,"","revert_tip_cells"]],"mocafe.angie.tipcells.TipCellsField":[[16,2,1,"","add_tip_cell"],[16,2,1,"","compute_phi_c"],[16,2,1,"","eval"]],"mocafe.expressions":[[17,1,1,"","EllipseField"],[17,1,1,"","EllipsoidField"],[17,1,1,"","PythonFunctionField"],[17,1,1,"","SmoothCircle"],[17,1,1,"","SmoothCircularTumor"]],"mocafe.fenut":[[19,0,0,"-","fenut"],[20,0,0,"-","log"],[21,0,0,"-","mansimdata"],[22,0,0,"-","parameters"],[23,0,0,"-","solvers"]],"mocafe.fenut.fenut":[[19,3,1,"","build_local_box"],[19,3,1,"","divide_in_chunks"],[19,3,1,"","flatten_list_of_lists"],[19,3,1,"","get_mixed_function_space"],[19,3,1,"","is_in_local_box"],[19,3,1,"","is_point_inside_mesh"],[19,3,1,"","load_parameters"],[19,3,1,"","setup_pvd_files"],[19,3,1,"","setup_xdmf_files"]],"mocafe.fenut.log":[[20,1,1,"","DebugAdapter"],[20,1,1,"","InfoCsvAdapter"],[20,3,1,"","confgure_root_logger_with_standard_settings"]],"mocafe.fenut.log.DebugAdapter":[[20,2,1,"","process"]],"mocafe.fenut.log.InfoCsvAdapter":[[20,2,1,"","process"]],"mocafe.fenut.mansimdata":[[21,3,1,"","save_sim_info"],[21,3,1,"","setup_data_folder"]],"mocafe.fenut.parameters":[[22,1,1,"","Parameters"],[22,3,1,"","from_dict"],[22,3,1,"","from_ods_sheet"]],"mocafe.fenut.parameters.Parameters":[[22,2,1,"","as_dataframe"],[22,2,1,"","get_value"],[22,2,1,"","is_parameter"],[22,2,1,"","is_value_present"],[22,2,1,"","set_value"]],"mocafe.fenut.solvers":[[23,1,1,"","PETScNewtonSolver"],[23,1,1,"","PETScProblem"]],"mocafe.fenut.solvers.PETScNewtonSolver":[[23,2,1,"","solver_setup"]],"mocafe.fenut.solvers.PETScProblem":[[23,2,1,"","F"],[23,2,1,"","J"]],"mocafe.litforms":[[25,0,0,"-","prostate_cancer"],[26,0,0,"-","xu16"]],"mocafe.litforms.prostate_cancer":[[25,3,1,"","df_dphi"],[25,3,1,"","prostate_cancer_chem_potential"],[25,3,1,"","prostate_cancer_form"],[25,3,1,"","prostate_cancer_nutrient_form"]],"mocafe.litforms.xu16":[[26,3,1,"","xu2016_nutrient_form"],[26,3,1,"","xu_2016_cancer_form"]],"mocafe.math":[[27,3,1,"","estimate_cancer_area"],[27,3,1,"","estimate_capillaries_area"],[27,3,1,"","shf"],[27,3,1,"","sigmoid"]],mocafe:[[17,0,0,"-","expressions"],[27,0,0,"-","math"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function"},terms:{"0":[0,2,3,5,6,7,8,9,11,15,16,17,26,27],"000":[2,3,5,6,7,8,9],"0000":[7,21],"0001":[7,21],"001":[5,7,8,9],"0019989":[0,13,15,16],"009":0,"01":[7,8,9],"0149422":[0,26],"03":0,"04":11,"0e6":[7,8,9],"1":[0,2,3,4,5,6,8,9,14,15,16,17,19,26,27],"10":[0,6,11,13,15,16,25,26],"100":[0,2,3,6,7,8,9,17,27],"1000":[7,8,9],"1003":[7,8,9],"1007":0,"1016":0,"1058693490":0,"1073":[0,25],"11":[0,2,3,6,12,13,15,16,26],"1145":0,"11588":0,"12":[],"130":[7,8,9],"1371":[0,13,15,16,26],"14":[0,1,5],"15":[0,5,10],"150":[7,8,9],"16":[0,7,8,9,25],"1615791113":[0,25],"17":[0,8],"2":[0,2,3,4,5,7,8,9,11,14,15,16,19,26],"20":11,"200":3,"2000":[8,9],"2011":[0,2,3,6,13,15,16],"2013":0,"2014":0,"2015":0,"2016":[0,8,9,25,26],"2017":0,"2019":[11,19],"2021":0,"20553":0,"21":[],"225":[],"23":[],"2368":[],"239":[],"25":5,"2566630":0,"27s_laws_of_diffus":0,"2c":6,"2d":[2,3,5,6,7,8,9,14,16],"3":[0,2,3,11,14,16,19],"30":5,"300":[2,7],"319":0,"32":5,"33287":[],"34":6,"365":[7,8],"37":2,"375":2,"3c":6,"3d":[4,5,6,13,14,15,16],"4":[2,3,7,8,9,11,15,16,19],"400":7,"4c":6,"4d":6,"4th":15,"5":[0,2,5,7,8,9,13,15,16,19,26],"500":9,"512":[],"515":0,"52461":0,"52462":0,"548":0,"6":[0,13,15,16,19,26],"600":[7,8,9],"625":6,"642":[],"6_9":[],"6e5":[7,8,9],"7":[0,6,19],"73":[7,8,9],"75":[7,8,9],"8":[3,19],"9":[0,6],"961":[7,8,9],"978":0,"\u00e1":0,"\u00e9":0,"\u00f8lgaard":0,"aln\u00e6":0,"boolean":2,"case":[2,3,5,8,9,11],"class":[2,5,8,13,14,16,17,18,19,20,22,23],"default":[2,5,8,11,12,19,21,27],"do":[2,3,5,6,7,8,9,11,16],"final":[2,5,6,8],"float":[17,19,21,27],"function":[1,2,5,6,7,8,9,13,15,16,17,19,25,26,27],"hern\u00e1ndez":[13,15,16],"import":[2,3,5,6,7,8,9,11],"int":[2,3,6,13,16,19,23],"long":19,"new":[2,5,7,8,9,16,21,22],"poir\u00e9":[13,15,16],"public":[6,25,26],"return":[2,13,14,15,16,19,20,21,22,25,26,27],"short":[2,3,8,9],"true":[2,3,6,7,9,13,16,19,21,22],"try":[6,11],"var":11,"while":[2,3,8],A:[0,2,4,6,7,8,9,13,15,16,19,25,26],And:[2,5,6,7,8,9,11],As:[2,3,5,7,8,9],At:7,By:12,For:[2,3,5,6,7,8,13,15,16,21,25,26],If:[2,3,5,8,9,11,13,15,16,19,21,25,26],In:[1,2,3,5,6,7,8,9,10,11,15,16,19,27],It:[5,13,22,25],ONE:[0,13,15,16,26],Of:[3,8,9,11],One:[2,3,5,9],The:[0,2,3,6,7,8,9,10,11,13,15,16,17,19,20,21,22,25,26],Then:[2,3,5,6,7,8,9,11],There:11,These:[2,8],To:[2,3,5,6,7,8,11],Will:21,With:[2,3,8,9],_:[],__file__:[2,3,6,7,8,9],__init__:[],__name__:[3,7,8,9],_multipl:[],a_valu:7,abh:[0,5,10],abl:[3,5,6,8,9],about:[5,22],abov:[2,3,5,6,7,8,9,11,16],academi:[0,25],access:[0,5,13,24],accord:[2,5,6,8,16,26],accordingli:[2,3,6,7,8,9],account:2,acm:0,across:[2,5],act:[],action:[6,7],activ:[2,3,6,16],activate_tip_cel:[2,3,6,16],actual:[2,3,5,8],ad:[2,3,15,21],adapt:[6,7,20],add:[11,16],add_tip_cel:16,addit:2,addition:2,additionali:8,adimdiment:[7,8,9],adimension:[7,8,9],adiment:[7,8,9],advanc:5,advantag:[8,11],af:[2,3,6,13,15,16],af_0:[2,3,6,15],af_:2,af_at_point:16,af_c:2,af_expression_funct:[],af_p:[2,15,16],af_sourc:[1,2,3,6,12],afexpressionfunct:[],after:[2,3,7,8],again:[2,3,5,9],agent:[12,13,14],agreement:8,aim:[],al:[2,3,6,8,13,15,16,26],algebra:[5,7,8,23],algorithm:[2,5,12,14],all:[2,3,4,5,6,7,8,11,12,16,25],allow:11,almost:19,alnaeslolgaard:[0,1,5],along:[2,8],alpha:15,alpha_p:[2,6,15,16],alpha_t:[2,15],alreadi:[2,3,8,11],also:[2,3,4,5,6,8,10,11,13,16,25,26],altern:[],alwai:[3,9],amg:3,among:[],an:[0,2,3,5,6,7,8,9,10,12,13,17,21,22],analyz:8,ander:0,angi:[1,2,3,6,10],angiogen:[2,6,13,15,16],angiogenesi:[0,1,4,6,10,12,13,15,16,26],angiogenesis_2d:[2,6],angiogenesis_2d_initial_condit:[],angiogenesis_3d:3,angiogenesis_form:[2,3,6,15],angiogenic_factor_form:[2,3,6,15],ani:[2,3,5,8,9,11,13,22],anim:5,anoth:2,anymor:11,apoptosi:[8,25],append:[],appli:[0,2,3,8,13,16],applic:0,apply_sourc:[2,3,6,13],appreci:[],approach:12,appropri:[3,9],approxim:[2,5],apt:10,ar:[2,3,5,6,7,8,9,12,13,15,16,19,21,22],archiv:0,area:27,arg0:23,arg1:23,arg2:23,arg3:23,arg:20,argument:[2,5,6,7,8,15,20,21,25,26],argv:8,around:[],arrai:[7,8,9],articl:0,as_datafram:22,ask:[2,7,21],asm:[],aspect:[6,18],assign:[2,3,5,6,7,8,9],associ:25,assum:8,august:0,author:[6,8],auto:21,auto_enumer:[2,3,6,7,8,9,21],automat:[2,7,8,11,21],autoremov:11,auxiliari:[2,3,15],auxilliari:2,avail:[6,11],averag:8,avoid:8,b:[0,2,3,7,8,9,19,27],backend:[5,23],backward:[5,25,26],ban:[],bar:[2,3,6,7,8,9],base:[8,10,14,26],base_class:[1,12],base_loc:[],basecel:[13,14],bash_complet:11,basi:[15,17],basic:[6,7,22],bc:23,becaus:[2,5,8,9,11],becom:2,been:[3,11],befor:[2,3,5,8],begin:[5,8],behav:2,behaviour:[2,5],being:[],below:[1,2,3,5,6,7,8,9,11,15,19],bengzon:[],benjamin:0,berlin:[],best:[5,8,9,19],better:[8,10],between:[8,27],beyond:[],bibliographi:10,bigger:19,bin:11,bind:11,biocomputingup:11,biolog:[],bit:[2,8],black:5,blechta:0,block:[],blood:[3,6,13],book:5,bool:21,border:[17,19],border_width:19,bot:[],both:[2,3,5,8,9,14,20],box:[5,11,19],boxmesh:[3,9],brief:4,briefli:[2,5,15],bsymbol:[],bug:11,build:[11,19,25,26],build_local_box:19,built:[5,8,11],builtin:5,c0:15,c:[2,3,5,6,8,11,13,15,16,26,27],c_0:[2,3,6],c_0_exp:[2,6],c_c:2,c_exp:3,c_x:5,c_y:5,cach:11,cahn:[2,15],cahn_hillard_form:15,calcul:[],call:[2,5,6,7,8,15,16,20,21,23],came:5,can:[1,2,3,5,6,7,8,9,10,11,15,16,17,19,20,21],cancer:[0,1,2,4,7,10,24,25,26,27],cannot:2,canon:5,capillari:[2,8,15,16,26,27],capilllari:2,care:[2,3,5,9,13],carefulli:6,carlo:0,castro:[0,13,15,16],caus:2,cc:[],cdot:[2,5,7,8,15,16,27],cell:[2,3,6,8,13,14,16],cell_radiu:13,center:[2,5,7,8,9,13,16,17,27],central:27,certain:2,cg1_element:8,cg:[2,3,6,7,8,9,19],cgi:0,chang:[2,3,6,7,8,9,10,11],chapter:[],check:[2,3,5,7,8,9,13,14,16,19,22],checker:13,cheker:[],chem_potenti:15,chem_potential_const:25,chemic:[5,25],chempot_const:[7,8,9,25],chi:[2,6,7,8,9,16,25],chi_valu:7,choic:[3,5,9,19],choos:[5,8,9],chose:8,chosen:8,chri:0,chunk:19,circl:[2,3,5,13,16,17],circle_radiu:13,circular:17,cite:[2,5,13,15,16,25,26],clarif:2,clarifi:8,clean:11,clear:[],clearli:[5,6],click:[2,3,4,5,6,7,8,9],clock:13,clock_check:13,clockcheck:13,close:5,closer:2,closest_tip_cel:16,cluster:11,cma:0,coarsen:0,code:[4,10,16,21],coher:16,collabor:[3,8,9,12,25,26],collaps:[2,3,6,7,8,9],collect:5,column:22,com:11,come:2,comm:[2,3,6,7,8,9,19,23],comm_world:[2,3,6,7,8,9,19],command:[2,8,11,21],comment:8,common:[8,11,15,25],commun:[11,19],compar:[3,6],compil:11,complet:[1,2,5,6,7,8,10,25,26],complex:2,compon:8,compos:[2,8,19],compromis:[],compulsori:22,comput:[0,3,5,6,8,9,11,16,19],compute_phi_c:16,compute_tip_cell_veloc:16,conacyt:0,conatin:[],concentr:[2,5,8,15,16],condit:[6,7,13,15,16,26],conductor:5,confgure_root_logger_with_standard_set:20,confid:[],configur:[8,9,20],conserv:19,consid:[2,3,5,8,9,11,15],constant:[2,3,5,6,8,15,16,25,26],constant_valu:[],constantafexpressionfunct:[],constantli:[2,8],constantsourcesfield:13,constrain:[3,9],construct:[5,8],constructor:2,consum:[8,26],consumpt:[2,15],contain:[2,6,7,8,10,12,13,16,18,19,21,22,24,25,26],content:[],contextu:20,continu:[2,8],continuo:19,contributor:0,conveneinc:15,conveni:[2,6,7,8],coordin:2,copyright:[],core:[8,22],correct:6,correctli:11,correspond:[15,22,25,26],correspong:25,corvera:0,costant:15,could:[2,15],count:2,coupl:[0,8,26],cours:[3,5,8,9,11],cpp:[13,15,16,19,23,25],cpu:[2,3,6,7,8,9],creat:[2,3,5,6,7,8,9,11,16,21,22],creation_step:[13,14,16],credit:[],critic:[5,6],crop:6,cst:[],csv:20,cube:5,current:[2,5,6,7,8,9,11,13,16,19,21],current_step:[7,8,9,16],cxx:[],cylinder_radiu:3,cylindr:3,d:[0,2,3,4,5,6,8,11,13,15,16],d_:[2,26],d_sigma:26,data:[1,2,3,4,5,9,11,18,21],data_fold:[2,3,6,7,8,9,19,20,21],datafram:22,date:21,dateandtim:21,dc:15,de:0,deactiv:[2,16],debug:20,debugadapt:20,decai:[8,25],decemb:0,decid:[5,21],decreas:2,def:[6,7,11],defin:[1,2,3,5,6,7,8,9,15,16,19,20,23,25,27],definit:[7,11,15],defint:2,degre:[2,3,5,6,7,8,9,19],deleg:[],deliber:5,delta:[7,8,9,25],demo:[2,3,5,6,7,8,9,10,13,15,16],demo_doc:[],demo_doc_jupyt:4,demo_doc_python:4,demo_in:[2,3,6],demo_out:[2,3,5,6,7,8,9],denser:6,depend:[2,5,8,11],deprec:19,deriv:[2,5,6,8,15,25,27],desc:[2,3,6],describ:[2,3,5,8,9,13,15,16,25,26],descript:[6,25,26],desid:19,design:3,desir:6,desktop:[3,9],detail:[5,15],determin:17,develop:[5,6,8,10],df:[8,15],df_dphi:25,dict:[6,21,22,23,26],dictionari:[6,8,22],did:[2,3,8,9],differ:[2,3,5,6,7,8,9,16,18,19],differenti:[0,3,5,8,9],difficult:[],diffus:[0,2,8,15,25,26],dimens:[3,8,9,14],dimenson:[],directli:23,directori:21,discours:[],discret:[7,13,14,15,25,26],discuss:[2,5,8,15],displai:[6,7],distanc:[2,14,16],distant:2,distibut:[],distribut:[2,5,8,13,14,15,19],divid:[2,19],divide_in_chunk:19,divis:19,docker:11,document:[5,8,11,25],doe:[],doesn:[],doi:[0,13,15,16,25,26],dokken:5,dolfin:[13,15,16,19,23,25,26,27],domain:[0,2,5,8,9,13,16],don:[2,3,7,8,9,11],done:[8,9],dot:5,doubl:[7,8,9,25,26],down:8,download:[2,3,4,5,6,7,8,9,11],driven:[2,8],drop:2,dt:[2,3,5,6,7,8,9,15,25,26],due:6,duplic:8,dure:[2,3,6,7,8,9,21],dx:5,dynam:2,e0149422:[0,26],e19989:[0,13,15,16],e:[0,2,5,6,8,13,15,16,27],each:[2,5,6,7,8,13,15,16,19,21],eas:[3,9],easi:[2,3,6,7,8,9],easier:[2,6,7,11],easiest:8,easili:[2,8,11],easli:[2,3,6,7,8,9],east:13,effect:2,effici:[3,5,8],effort:[3,9],egg:11,either:20,elabor:19,element:[2,5,8,9,15,19],element_typ:19,elimin:[],ellips:[9,17],ellipsefield:[7,8,17],ellipsoid:[9,17],ellipsoidfield:[9,17],ellipt:8,elment:5,els:[2,3,6,7,8,9,16],empti:8,en:0,encount:11,encourag:5,encyclopedia:0,end:7,endotheli:[2,6],energet:15,engin:0,enough:[],ensur:5,enter:[],entir:[2,5,8],epsilon:[2,7,8,9,15,25],equal:[2,13,19,21,25],equal_for_all_p:[],equat:[0,3,8,9,15,25,26],error:[2,3,6,7,8,11,21],error_messag:6,error_msg:[6,21],es:0,especi:21,espress:[],estim:[8,27],estimate_cancer_area:27,estimate_capillaries_area:27,et:[2,3,6,8,13,15,16,26],etc:[2,11,18],eugenia:0,euler:[5,25,26],ev:14,eval:16,evalu:16,even:[3,6,8,9,19],everi:[5,8,16],everyth:[2,3,8,11,22],everywher:2,evid:6,evidenc:6,evolut:[5,8,26],exactli:[2,3,6,8,9],exampl:[4,5,13,15,16,19],excel:[5,8],except:[2,3,6],execut:[6,7,11,21],execution_tim:[6,7,21],exist:[2,3],expect:2,experi:8,experienc:[5,8],expern:[2,3,8,9],explain:[2,5,11],explan:[6,7],exploit:[2,3,6,7,8,9],express:[1,2,3,5,6,7,8,9,10,13,16],extens:[2,5,8],extra:20,extrem:[2,3,5],ey:[],f77:[],f90:[],f95:[],f:[2,3,6,7,8,9,15,23],fact:5,factor:[2,6,13,15,16],fals:[2,3,6,8,9,13,16,19,22],familiar:[],far:2,faster:[],fct:0,featur:[5,8],feb:0,feel:5,fem:[2,5,8],fenic:[0,1,2,3,4,6,7,8,9,10,12,13,15,16,19,25,26],fenicsproject:[],fenicsx:[5,11],fenut:[1,2,3,6,7,8,9,10,13,15,16,25,26],few:[5,8],ffor:20,fibonacci:14,fibonacci_spher:14,fick:[0,2,8],field:[0,1,2,3,4,6,7,8,9,10,12,13,15,16,17,21,24,25,26,27],fig2c:6,fig2d:6,fig3c:6,fig3d:6,fig4c:6,fig4d:6,fig:6,figur:5,file:[2,3,5,6,7,8,9,11,18,19,20,21,22],file_af:[2,3,6],file_c:[2,3,6],file_fold:[2,3,6,7,8,9],file_nam:[2,3,6,19],find:[1,2,3,5,6,8,9,10,11,15],finger:7,finit:[2,5,9,15],finiteel:8,fir:[],first:[2,3,5,6,7,8,11,21],flat:19,flatten:19,flatten_list_of_list:19,flow:5,fluid:5,fo:25,folder:[2,3,6,7,8,9,11,19,20,21],folder_nam:21,folder_path:[2,3,6,7,8,9,21],follow:[1,2,3,5,6,7,8,9,10,11,15,16,21],follw:15,form:[0,1,3,6,7,9,12,22,25,26],form_af:[2,3,6],form_ang:[2,3,6],format:8,formul:0,formula:[5,16],forum:8,found:16,fpradelli94:[],frac:[2,5,8,15,16,27],frame:22,fredrik:[],free:[0,5],frequenc:0,from:[2,3,5,6,7,8,9,11,13,14,20,22,26],from_dict:[7,8,9,22],from_ods_sheet:[2,3,6,22],fu:[2,3,6],full:[1,11,13],fun:[2,3,6],function_spac:[2,3,6,7,8,9],functionspac:[5,8,27],further:15,futur:11,g:[0,2,5,6,7,8,9,16,25,26],g_m:[2,16],galerk:8,galleri:[2,3,5,6,7,8,9,10],gamg:[3,7,8,9],gamma:[5,7,8,9,25],garth:0,gcc:[],ge:2,gear:5,gener:[2,3,4,5,6,7,8,9,13,15,19,21,26],genericmatrix:23,genericvector:23,get:[2,3,5,6,7,11,13,14,16,22],get_dim:[],get_dimens:14,get_dist:14,get_global_mesh:[],get_global_source_cel:13,get_global_tip_cells_list:16,get_latest_tip_cell_funct:[2,3,6],get_local_mesh:[],get_local_source_cel:13,get_mixed_function_spac:[2,3,6,7,8,9,19],get_point_value_at_source_cel:[],get_posit:14,get_radiu:16,get_rank:[2,3,6,7,8,9],get_valu:[2,3,6,7,8,9,22],gif:[],git:11,github:11,githubusercont:11,give:5,given:[2,3,5,7,8,9,13,14,16,17,19,21,22,27],given_list:19,given_valu:2,glaerkin:19,global:[13,16],gmre:[3,7,8,9],go:[5,6,7,8,11],goe:8,gomez:[0,25,26],good:5,grad:[2,3,5,6],grad_af:[2,3,6,16],grad_af_function_spac:[2,3,6],grad_t:[2,3,6],gradient:[2,15,16],gradual:5,gram:[7,8,9],great:[3,9],greatest:5,group:[],grow:2,growth:[0,2,8,25,26],guarante:9,guid:11,guillermo:0,gulbenkian:0,h5:7,h:[0,15,25,26],ha:[3,5,11,13,16,17],had:11,hake:0,han:0,hand:2,handl:[2,15],happen:2,hard:8,have:[2,3,5,6,7,8,9,11,21,22,27],healthi:26,heavisid:15,heavysid:27,hector:0,heidelberg:0,help:11,here:[2,3,5,6,7,8,9,11],hern:0,hi:3,hierarch:0,high:[2,5,6],higher:[2,3,9],hillard:[2,15],home:11,host:[2,11],how:[4,5],howev:[2,3,5,6,7,8,9,11,15],hpc:[3,9,11],html:[7,21],http:[0,11,13,15,16,25,26],hugh:[0,25],hurri:[],hybrid:12,hydra:[],hypothesi:8,hypox:[2,13,26],i:[2,6,8,11,13,16],i_v_w:[2,6],id:0,ident:[3,8,9],identifi:[19,21],ignor:[],imag:[6,11],implement:[5,10,15,16],implment:4,imput:[],includ:[2,8,10,11],inde:[2,3,5,7,8,9,15,21],index:[0,10],induc:[2,13],inf:27,influenc:6,info:[18,20,21],infocsvadapt:20,inform:[2,8,20,22],inherit:14,inhibit:2,init:[3,7,8,9],init_tim:[6,7],initi:[6,7,15,25,26],initial_vessel_radiu:3,initial_vessel_width:[2,3,6],input:[2,5,21],input_mesh:[],insert:20,insid:[2,3,5,7,8,11,16,17,19,22],inside_valu:[7,8,9,17],inspect:[8,10],instal:[2,3,6,7,8,9,10],instanc:[2,7,21,22],instead:[3,8,9,11,19],instruct:11,int_:5,integr:10,interact:2,interest:[2,5,7,8,19],interfac:[0,2,3,5,8,26],intern:[16,20,23],interpol:[2,3,6,7,8,9],intracomm:[19,23],introduc:2,introduct:4,introduction_to_fen:5,invit:5,involv:[2,15],io:[],ipynb:[2,3,5,6,7,8,9],ipython:[],is_in_ellipse_cpp_cod:[],is_in_local_box:19,is_inside_global_mesh:[],is_inside_local_mesh:[],is_paramet:22,is_point_insid:16,is_point_inside_mesh:19,is_value_pres:22,isbn:0,isciii:0,iter:[2,3,5,8,9],its:[2,8,11,15,16,20,25],itself:[2,8,15,20],j:[0,2,3,5,6,7,8,9,13,15,16,23,25,26],j_mat:[3,7,8,9],jacobian:[2,6,8],jan:0,jessica:0,jiangp:0,job:[2,8],johan:0,johann:0,johansson:0,jour:[],journal:[0,13,15,16,26],json:19,juan:0,jun:0,jupyt:[2,3,4,5,6,7,8,9],just:[2,3,4,5,6,7,8,9,11,13,21],k:[0,25],keep:[2,13],kehlet:0,kei:[2,8],kevin:0,keyword:20,kind:[5,15],know:5,known:8,kristian:0,krylov:[],ksp_type:[3,7,8,9],kwarg:[13,20],l:[5,7,25],la:23,lagrang:[5,15],lambda:[2,3,7,8,9,13,15,25],lambda_:26,lambda_phi:26,langtangen:0,langtangenlogg2017:5,languag:[0,1,5,11],laptop:[3,9],larg:[8,11],larson:[],last:[2,5,11],later:[],latest:[],latter:[2,15],law:0,lb13:[],le:[2,15,16],lead:[2,5,6,7,15],least:22,leav:8,left:[2,6,8],legaci:[],lei:0,length:[],less:6,let:[2,5,8],level:[2,3,5,6,20],leverag:[5,8],libexec:11,librari:8,licens:[],like:[4,5,7,8,11],limit:2,line:[2,5,8,21],linear:[2,8,23],linearli:2,link:[2,3,6,7,8,9,11],linux:11,list:[7,8,11,13,16,19,25],list_of_list:19,liter:[7,8,9],litform:[1,7,8,9,10],liu:[0,25],ll17:[0,5],ll:[3,9,20],lmbda:15,load:[2,3,6,8,9,22],load_paramet:19,load_random_st:[],loading_messag:[6,7],local:[2,11,13,19],local_box:19,local_mesh:19,locat:9,log:[1,2,3,6,8,18],logg:0,logger:[5,20],loggeradapt:20,loglevel:[2,3,6,7,8],longer:3,look:[5,6,8,11],loop:[5,6,7],lorenzo:[0,8,9,25],low:[2,6,8],lower:[6,9,15],lst:[0,7,8,9,25],lx:[2,3,6],ly:[2,3,6],lz:3,m:[0,2,8,9,11,13,15,16,25],m_:26,m_c:2,m_phi:26,machado:[0,13,15,16],made:3,magnitud:[3,9],mai:19,main:[2,8,11,15],mainli:[5,20],make:[2,3,5,6,7,8,9,15],manag:[1,2,3,13,16,18,21],mang:[],mani:6,manipul:20,mansimd:[2,3,6],mansimdata:[1,2,3,6,7,8,9,18],manzanequ:[0,13,15,16],map:[2,3,13],mar:0,mari:0,mario:0,martin:0,mat:[3,7,8,9],matemat:[],math:[0,1,2,3,8,9,10,15],mathemat:[0,5,10,13,15,16,26],matrix:8,matter:[7,8],max:[8,27],maximum:15,mcte:0,mean:[2,8],measur:[2,6,7,22],mechan:0,ment:5,mention:[2,5,8],merg:2,mesh:[3,6,7,13,16,19],mesh_dim:16,mesh_fil:3,mesh_wrapp:[],mesh_xdmf:3,meshwrapp:[],messag:[6,7,8,11,20,21],met:[13,16],meta:4,method:[0,2,5,6,7,8,11,13,15,16,18,19,20,21,25,26],metod:5,michael:0,micinn:0,might:[3,5,6,7,8,9,11,13,19],migth:[],min:8,min_tipcell_dist:16,minut:[2,3,5,6,7,8,9],mistak:[2,6,7],mix:[8,19],mixed_el:8,mixedel:8,mkdir:[2,3],moacf:11,mobil:26,mocaf:[1,4,5],mocafe_fold:[],mocef:[],model:[0,1,3,4,5,6,7,9,10,12,13,15,16,24,25,26],modifi:[2,20],modul:[8,10,12,13,16,19,20,24,25,26],moment:[],monitor:8,more:[2,5,8,16,19],moreov:2,most:[3,5,6,7,8,9],motion:[],move:[0,2,3,6,16],move_tip_cel:[2,3,6,16],mpar:[2,3,6],mpi4pi:[19,23],mpi:[2,3,6,7,8,9,10,13,16,19,21,23],mpi_comm:[],mpich:11,mpirun:[2,3,6,7,8,9,11],msg:20,mshr:[2,6],mu0:15,mu:[2,3,6,8,9,15],mu_0:[2,3,6],much:[],multidimension:[],multipl:21,multiple_angiogenesis_simul:6,multiple_pc_simul:7,must:[8,22,25,26,27],mv:[2,3],mx:0,my:[],my_sim_data:21,n:[0,2,3,5,6,7,8,9],n_chnk:19,n_global_mesh_vertic:[],n_local_mesh_vertic:[],n_point:14,n_sourc:[2,3,6,13],n_step:[2,3,6,7,8,9],n_variabl:19,nabla:[2,5,8,15,16],name:[2,6,8,19,21,22,25,26],name_param_a:22,name_param_b:22,nation:[0,25],natur:[2,3,5,8,9],ncol:[2,3,6,7,8,9],ndarrai:[13,14,17],ndez:0,nearbi:[6,7],necessari:[2,8],need:[2,3,5,6,7,8,9,11,19,20,21],nest:7,network:[2,6],neumann:[2,3,5,8,9],never:5,new_posit:16,new_valu:22,newton:[],newtonl:[3,7,8,9],newtonsolv:23,next:[2,5,7,8,9,11],nl:23,non:[8,13],none:[2,3,6,7,8,9,13,21,23],nonlinear:[8,23],nonlinearproblem:23,norm:[2,16],normal:[3,6,9,19,20],notch:2,note:11,notebook:[2,3,4,5,6,7,8,9],noth:[5,8,13,16,20,21],notic:[2,3,5,6,7,8,9,11,19],nov:[],now:[2,3,6,7,8,9,11],np:[7,8,9],number:[2,3,5,6,7,8,9,14,19],numer:[0,8],numpi:[7,8,9,11,13,14,17],nutrient:[2,8,25,26],nutrient_weak_form:[],nx:[2,3,6,7,8,9],ny:[2,3,6,7,8,9],nz:[3,9],object:[2,3,5,8,13,19,21,22,23,25,26],obscur:8,observ:6,obtain:5,occur:[2,8,21],occurr:[],od:[2,3,6,11],odf:22,off:[2,3,6],offici:11,often:[2,5,6,7],old:5,oldid:0,omega:[2,5,15],onc:[2,3,6,7,8,9],one:[2,3,5,6,8,9,15,16,19,20,21],onli:[2,3,6,8,14,16,20],onlin:[0,5],onward:11,open:[5,10,11],openmpi:11,oper:[2,5,7,8],optim:[8,11],option:[2,3,6,7,8,9,11,13,21],order:[2,3,5,7,8,9,15,19,21,27],ore:[25,26],org:[0,11,13,15,16,25,26],origin:[2,5,7,8,13,15,16,25,26],other:[2,5,6,7,8,11,13,16,22],otherwis:[2,13,16,19,21,22],our:[2,5,8],out:[2,3,7,8,9,11],outlin:8,output:[6,7,11],outsid:[5,8,17],outside_valu:[7,8,9,17],over:19,overrid:20,own:[8,20],oxygen:2,p0:[],p:[0,15,16,26],packag:[5,8,10,11],page:[2,3,6,7,8,9,10,11],panda:[11,22],paper:[2,7,8,13,15,16,25,26],parallel:[2,3,6,7,8,9,10,19,21],param:[6,19,26],param_df:22,param_dict:22,paramet:[1,2,3,6,7,8,9,10,13,14,15,16,18,19,20,21,25,26,27],parameters_fil:[2,3,6,19],parameters_to_chang:6,paraview:[5,6,7],parent:[2,3,6,7,8,9],part:[2,13],partial:[0,2,5,8,15],particular:[0,6],pass:[2,20],path:[2,3,6,7,8,9,11,20,21,22],pathlib:[2,3,6,7,8,9,20,21,22],pathwai:2,pattern:[0,8,13,15,16],pbar:[2,3,6],pc:7,pc_model:[7,8,9],pc_type:[3,7,8,9],pde:[0,2,5,8],peopl:5,perfectli:5,perform:[2,8],person:[0,25],perspect:[],petsc4pi:[3,7,8,9],petsc:[3,7,8,9,23],petscmatrix:[3,7,8,9],petscnewtonsolv:23,petscproblem:[8,23],petscvector:[3,7,8,9],petter:0,phase:[0,1,2,3,4,6,7,8,9,10,12,13,15,16,17,24,25,26,27],phi0:[5,7,8,9],phi0_cpp_cod:[],phi0_in:[7,8,9],phi0_max:[],phi0_min:[],phi0_out:[7,8,9],phi:[5,7,8,9,25,26,27],phi_0:26,phi_c:16,phi_prec:25,phi_th:16,phi_xdmf:[5,7,8,9],philosophi:3,php:0,physic:5,pi:[2,16],pick:[2,8],pictur:8,piec:[5,11],pip3:11,pip:10,place:[2,3,5,8,13,19,20],plan:[],platform:[4,5,11],pleas:[8,25,26],plo:[0,13,15,16,26],ploson:0,pna:[0,8,25],png:[],point:[0,2,3,5,6,7,8,9,13,14,16,19],poir:0,polynomi:3,pone:[0,13,15,16,26],popul:2,popular:5,portabl:[8,11],posit:[2,3,6,13,14,16,19],possibl:[3,5,6,9,15,16],post:8,potenti:[15,25,26],pow:[3,5],power:[3,5,9],ppa:11,practic:[2,5],precis:[8,16,19],precondition:[],preconditioner_typ:3,prefer:[11,19,23],presenc:[5,8],present:[2,6,7,8,9,12,13,16,22,25],preserv:6,pretti:[8,26],previou:2,print:[8,11],probabl:6,problem:[0,3,7,8,9,11,23],proce:[2,3,11],procedur:[11,16],proceed:[0,25],process:[2,3,6,7,8,9,13,16,19,20],produc:[],product:[6,26],progress:[2,3,6,7,8,9],progress_bar:[7,8,9],project:[0,2,3,5,6],prolif:26,prolifer:[2,6,8,15,16,25],properli:11,properti:11,prostat:[0,2,4,7,25],prostate_canc:[1,7,8,9,24],prostate_cancer2d:8,prostate_cancer3d:9,prostate_cancer_2d:8,prostate_cancer_3d:9,prostate_cancer_chem_potenti:25,prostate_cancer_form:[7,8,9,25],prostate_cancer_nutrient_form:[7,8,9,25],prostate_cancer_weak_form:[],provid:[2,3,4,5,6,7,8,10,11,13,16,19,21],pseudo:16,pt:0,publish:8,purpos:[2,8,10,11,16,19,20],put:[5,22],pvd:19,py:[2,3,5,6,7,8,9],pyhton:[],python3:[2,3,6,7,8,9,11],python:[0,2,3,4,5,6,7,8,9,10,11,17,20,22],python_fun:17,python_fun_param:17,pythonfunctionfield:17,q:15,quad:[2,5,15,16],quai:[],qualit:6,quick:11,quit:2,r:[0,5,13,15,16,25,27],r_c:[2,16],r_v:3,radiu:[2,13,14,16,17],rand_max:[7,8,9],random:[6,7,8,9],random_sources_domain:[2,6],randomli:[2,13,16],randomsourcemap:[2,3,6,13],randomst:[],rang:[2,3,5,6,7,8,9],rank:[2,3,6,7,8,9],rate:[2,6,8,15,16,25,26],rational:21,raw:11,re:[2,3,5,8],reach:[],read:[2,3,5,8,15,16,27],reader:11,readi:[8,11],real:8,realli:[3,8,9],reason:[2,3,6,7,9],recommend:[2,3,5,6,7,8,9,11],rectangl:[2,5,6],rectanglemesh:[2,6,7,8],reduc:[2,8,9],reduct:6,refer:[2,6,8,22,25,26],refin:0,regard:[5,8],rel:[2,19],relat:[1,8,15],releas:11,reload:3,relro:[],remain:[2,3,9,15],remaind:19,remark:9,rememb:[2,3,9,13,15,16,25,26],remov:[2,13,16],remove_global_sourc:13,remove_sources_near_vessel:[2,3,6,13],repeat:[6,7],report:[2,7,8,11,15,16,22,25],repositori:11,repres:[2,5,8,13,14,15,16,17,19,22,26,27],represent:5,reproduc:[2,8,10,11],requir:[2,3,5,8,9,11,19,25,26],research:[13,15,16,25,26],resembl:[3,9],resolv:[2,3,6,7,8,9],respect:[2,6,8,13],respons:[2,13],result:[5,19,21],retriev:[2,8],revert:[2,3,6],revert_tip_cel:[2,3,6,16],revis:2,rf:11,richardson:0,right:[2,6,16],rim:26,ring:0,risk:2,rm:11,rmw:[],rodrguez:[0,13,15,16],rogn:0,root:[2,20,21],rotational_expression_function_paramet:[],rotationalafexpressionfunct:[],round:7,routin:8,row:5,rui:0,rule:2,run:[5,11],run_angiogenesis_simul:6,run_prostate_cancer_simul:7,runtimeerror:6,s:[0,2,3,5,7,8,9,25],s_:8,s_av:[7,8,9],s_averag:[7,8,9],s_exp:[7,8,9],s_express:[],s_max:[7,8,9],s_min:[7,8,9],salt:5,same:[2,3,6,7,8,9,12,15,19,21],save:[2,3,4,5,8,9,21],save_random_st:[],save_sim_info:[6,7,21],saw:[6,7],scalar:15,scale:[0,8,25],scienc:[0,25],scientif:[5,8,25,26],scott:[0,25],screenshot:6,script:[2,3,4,5,6,7,8,9,10,11],search:10,second:[2,3,5,6,7,8,9,15,21],section:[2,5,6,7,11],secur:[],see:[2,3,4,5,6,7,8,9,11,13,15,16],seen:2,select:[2,16],self:[2,8,23],semi:5,semiax:[8,17],semiax_i:[7,8,9,17],semiax_x:[7,8,9,17],semiax_z:[9,17],separ:[2,3,15,26],set:[2,3,5,6,7,8,9,17,20,21,22],set_descript:[3,6,7],set_equal_randomstate_for_all_p:[],set_log_level:[2,3,6,7,8],set_valu:[6,7,22],setfromopt:[3,7,8,9],setfunct:[3,7,8,9],setjacobian:[3,7,8,9],setstat:[],setup:[6,7],setup_data_fold:[2,3,6,7,8,9,21],setup_pbar:3,setup_pvd_fil:19,setup_xdmf_fil:[2,3,6,7,8,9,19],sever:[6,8],shape:7,share:[],sheet:[2,22],shell:11,shf:27,should:[5,11],show:[2,3,5,6,8,9],shown:[2,6],shut:8,side:[2,3,5,8,9],sif:11,sigma0:[7,8,9],sigma0_in:[7,8,9],sigma0_out:[7,8,9],sigma:[7,8,9,25,26],sigma_0:26,sigma_old:25,sigma_xdmf:[7,8,9],sigmoid:[17,27],signal:2,significantli:8,sim2c:6,sim2d:6,sim3c:6,sim3d:6,sim4c:6,sim4d:6,sim_data:21,sim_descript:[6,7,21],sim_dict:6,sim_dict_kei:6,sim_info:[7,21],sim_nam:[6,7,21],sim_rational:[],sim_valu:22,similar:3,similarli:[2,8],simparam:[2,3,6],simpi:[],simpl:[2,5,8,27],simplest:5,simpli:[2,3,5,6,7,8,9,11,15,21,26],simplier:[],simul:[0,1,4,5,10,12,13,15,16,18,19,21,22,25,26],sinc:[2,5,8,11,16],singl:[2,5],singular:10,singularityc:11,situat:5,skip:[6,7],slightli:[6,9,16],slope:[17,27],small:3,smooth:[17,27],smoothcircl:17,smoothcirculartumor:17,sne:[3,7,8,9],snes_solv:[3,7,8,9],snes_typ:[3,7,8,9],snesproblem:[3,7,8,9],so:[1,2,3,5,6,7,8,11,15,21],softw:0,softwar:[0,2,3,5,8,9,11],solut:[2,3,7,8,9],solv:[0,2,3,6,7,8,9,15],solvabl:[2,15],solver:[1,3,7,8,9,18],solver_paramet:23,solver_setup:23,solver_typ:3,some:[4,5,8,11,15,19],someth:[8,11],sometim:5,soon:2,sourc:[2,3,4,5,6,7,8,9,10,11,13,14],source_cel:13,source_map:13,source_point:13,sourcecel:13,sourcecellsmanag:2,sourcecomput:[],sourcemap:[2,13],sources_in_circle_point:13,sources_manag:[2,3,6],sources_map:[2,3,6],sourcesfield:[],sourcesmanag:[2,3,6,13],space:[2,5,6,8,9,19,27],sparser:6,spatial:[7,13,16],speak:[5,8],specif:[0,3,4,5,8,9,15,20],specifi:[5,8,21,25,26],sphere:[3,14,16],sphinx:[2,3,4,5,6,7,8,9],sphinx_gallery_thumbnail_path:[],spiral:[],spline:0,split:[2,3,6,7,8,9,15],spread:14,springer:0,sprout:2,squar:[2,5,8,19],stabl:[],standard:[2,3,6,7,8,15,20],standerd:20,start:[0,2,3,5,8],start_point:13,starv:2,state:[],statement:[2,3],std_out_all_process:[2,3,6],std_paramet:[6,7],stem:2,step:[2,3,4,5,6,7,8,9,11,15,16,25,26],still:[6,7,8],stop:2,store:[2,3,5,6,7,8,9,19,21,22],str:[6,19,20,21,22],string:[5,6,7,19],strong:0,structur:[5,22],studi:[],stuff:8,sub:[2,3,6,7,8,9],subclass:20,submodul:1,subpackag:[1,12,18,24],subspac:2,subsystem:11,sudo:11,suffici:2,suggest:[2,8],suit:[8,20],sum:[2,15],summar:5,summari:5,superclass:[],suppli:[8,25],support:11,suppos:8,sure:[2,3,6,7,8,9],surpass:13,surround:13,surviv:2,sy:8,symbol:5,sympli:[],system:[2,5,6,7,8,11],t:[0,2,3,5,6,7,8,9,11,15,25,26],t_:6,t_c:16,t_valu:16,take:[2,3,6,7,8,9,11,13],tast:[],tau:[7,8,9,25],tc_p:16,team:15,techniqu:5,tell:[8,11],term:[2,5,8,15],termin:11,test:[2,3,6,7,15,25,26],test_condit:6,testfunct:[2,3,5,6,7,8,9,15,25,26],tew:[0,25],textrm:[2,5,15,16],than:[2,3,5,6,8,9,16,19],thei:[2,3,6,7,8,9,13],them:[2,3,7,8,9,10,11],theori:[],theta:[2,15],thi:[4,5,11,12,13,14,15,16,18,19,20,21,23,24,25,26],thicker:6,thing:[2,3,5,8,9],think:[],thinner:6,third:[2,5],thoma:0,thorugh:2,those:[5,8],though:[3,6,8,19],three:[2,6,7,11],threshold:[2,13],through:2,throughout:[2,8,16],thu:[2,3,6,7,9,16],time:[2,3,5,6,7,8,9,11,13,15,21,25,26],tip:[2,3,6,14,16],tip_cel:16,tip_cell_manag:[2,3,6],tip_cell_posit:16,tipcel:[1,2,3,6,12],tipcellmanag:[2,3,6,16],tipcells_field:[2,3,6],tipcells_xdmf:[2,3,6],tipcellsfield:16,tipcellsmanag:[2,3],tissu:[0,8,13,15,25,26],titl:0,todo:[],togeth:[2,5,7],told:[],too:[],took:2,tool:[6,7,16],toolkit:8,topic:5,total:[2,3,5,6,7,8,9,19],toward:8,tpoirec:[0,2,3,6,12,13,15,16],tqdm:[2,3,6,7,8,9,11],tradit:[],tran:0,transform:[],transit:11,translat:[5,8,13],travasso2011:[],travasso2011a:2,travasso:[0,2,3,6,12,13,15,16],tree:7,trial:[],trialfunct:5,triangl:8,trivial:[6,7],tue:[],tumor:[0,8,13,15,16,17,26],tumour:7,turn:[2,3,6],tutori:[2,3,5,8,9],two:[2,3,7,8,11,15,16],type:[5,8,11,19,21],u:[2,3,5,6,7,8,9,26],u_0:5,u_0_exp:5,u_xdmf:5,ub:0,ubuntu:11,ufl:[2,5,8,15,25,26],um:[7,8,9],under:21,understand:[5,10],uni:0,unifi:[0,1,5],uniform:8,uninstal:10,uniqu:[2,7],unit:[2,22],unitsquaremesh:5,unknown:15,up:[2,3,5,7,8,9,11],updat:[2,3,5,6,7,8,9,11,16],upload:[2,3,7,8,9],uptak:[25,26],url:0,us:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,19,20,21,22,23,25,26,27],usabl:5,user:[2,5,8,11,17,20,21],usernam:[],usr:11,usual:[2,19],util:[1,18],v1:[2,3,6,7,8,9,15],v2:[2,3,6,7,8,9,15],v3:[2,3,6],v:[2,5,15,16,25,26,27],v_:26,v_pc:26,v_uh:26,v_ut:26,valu:[2,5,6,7,8,9,15,16,17,22,25,26,27],value_param_a:22,value_param_b:22,var_phi:25,variabl:[2,3,5,8,15,19,25,26,27],variant:5,variat:[3,8],varibl:27,varphi:[8,9,25,26,27],vascular:[0,2,13,15,16],vascular_proliferation_form:15,ve:9,vec:[3,7,8,9],vector:[2,3,7,8,9,16],vectorfunctionspac:[2,3,6],vegf:2,veloc:[2,6,16],velocity_norm:16,veri:[2,5,6,7,8],verifi:11,versatil:26,version:[0,3,5,6,7,9,11],vessel:[2,3,6,13,27],view:0,vilanova:[0,25,26],visibl:6,visual:[5,19],vol:0,w:0,wa:[2,6,8,25],wai:[3,5,8,9,11,15,19,23],wander:[],wank:5,want:[2,5,7,8,11,19],warn:[],water:5,we:[2,3,5,6,7,8,9,11],weak:[0,1,3,7,9,15,25,26],weak_form:[2,3,6,7,8,9],websit:5,weight:15,well:[0,8,9,11,20,25,26],west:[],wget:11,what:[2,8,9,11],when:[2,6,13,16,19,21],where:[2,3,4,5,6,8,13,15,16,19,20,27],which:[2,3,5,6,7,8,9,11,13,15,16,17,19,22,26,27],who:5,why:[],width:[2,19],wikipedia:0,wikipediacontributors21:[0,2,8],window:11,wise:5,wish:8,without:[2,8,21],wl:[],won:11,wonder:[],word:5,work:[3,5,6,7,8,11,19,21,25,26,27],workflow:8,would:[6,7],wrap:2,wrapper:[13,22],wre:6,write:[2,3,5,6,7,8,9],written:5,wrong:8,wsl:11,www:[0,11],x:[2,3,5,6,8,16,27],x_max:[7,8,9],x_min:[7,8,9],xdmf:[2,3,5,6,7,8,9,19],xdmffile:5,xu16:[1,24],xu2016_nutrient_form:26,xu:[0,26],xu_2016_cancer_form:26,xvg16:[0,26],y:[7,11,25,27],y_max:[7,8,9],y_min:[7,8,9],year:[0,7,8,9],yongji:0,you:[1,2,3,4,5,6,7,8,9,10,11,13,15,16,19,20,21,25,26],your:[2,3,5,6,7,8,9,10,11,13,15,16,20,25,26],yourself:[5,8],youtub:[2,3,7,8,9],z:[],z_max:9,z_min:9,zero:[2,8],zhang:[0,25],zip:[4,7]},titles:["Bibliography","Code Documentation","Angiogenesis","Angiogenesis 3D","Demo Gallery","A brief introduction to FEniCS","Save simulations meta-data 2","Save simulations meta-data 1","Prostate cancer","Prostate cancer 3D","Welcome to Mocafe\u2019s documentation!","Installation","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.angie</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.angie.af_sourcing</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.angie.base_classes</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.angie.forms</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.angie.tipcells</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.expressions</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.fenut</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.fenut.fenut</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.fenut.log</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.fenut.mansimdata</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.fenut.parameters</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.fenut.solvers</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.litforms</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.litforms.prostate_cancer</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.litforms.xu16</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.math</span></code>"],titleterms:{"1":7,"2":6,"3":6,"3d":[3,9],"4":6,"function":3,A:5,In:[],The:[5,12,18,24],af_sourc:13,agent:2,angi:[12,13,14,15,16],angiogenesi:[2,3],apt:11,base:2,base_class:14,basic:5,below:[12,18,24],bibliographi:0,boundari:[2,3,5,8,9],brief:[2,5,8],can:[],cancer:[8,9],clockcheck:[],code:[1,2,3,5,6,7,8,9],complet:[],comput:2,condit:[2,3,5,8,9],contain:11,content:[2,3,5,6,7,8,9],data:[6,7],definit:[2,3,5,8,9],demo:4,differenti:2,diffus:5,discret:[2,5,8,9],document:[1,10,12,18,24],domain:3,each:[12,18,24],equat:[2,5],exampl:[2,3,6,7,8,9],experi:6,express:17,fenic:[5,11],fenut:[18,19,20,21,22,23],field:[],figur:6,find:[],follow:[],form:[2,5,8,15],full:[2,3,5,6,7,8,9,12,18,24],galleri:4,heat:5,how:[2,3,6,7,8,9],implement:[2,3,6,7,8,9],includ:[],index:[],indic:10,initi:[2,3,5,8,9],instal:11,introduct:[2,5,8],linux:[],litform:[24,25,26],log:20,manag:[6,7],mansimdata:21,math:27,mathemat:[2,8],mesh:[2,5,8,9],meta:[6,7],mocaf:[2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],model:[2,8],mpi:11,multipl:[6,7],note:[3,9],origin:6,packag:[],paper:6,paramet:22,paraview:[2,3,8,9],pde:[3,9],phase:[],pip:11,problem:5,prostat:[8,9],prostate_canc:25,provid:[12,18,24],recommend:[],remov:11,report:6,result:[2,3,6,7,8,9],run:[2,3,6,7,8,9],s:10,save:[6,7],setup:[2,3,5,8,9],simul:[2,3,6,7,8,9],singular:11,solut:5,solv:5,solver:23,space:3,spatial:[2,3,5,8,9],submodul:[12,18,24],summari:[],system:[3,9],tabl:[2,3,5,6,7,8,9,10],test:11,thi:[2,3,6,7,8,9],tipcel:16,uninstal:11,visual:[2,3,6,7,8,9],weak:[2,5,8],welcom:10,what:5,workflow:5,xu16:26,you:[],your:[]}})