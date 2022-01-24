def main():
    welcome_msg = """
                                                                
                                                                                                                    
                @@@@    @@@    @@@@                         
                @@@    @@@@    @@@                          
                @@@    @@@@    @@@                          
                 @@@    @@@@    @@@                         
                 @@@@    @@@    @@@@                        
                 @@@     @@@    @@@@                        
                 @@@    @@@     @@@                         
                                                            
          @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  @@@@             
          @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  @@@@@@           
          @@@@@ @@@@@@@@@@@@@@@@@@@@@@@@@   @@@@@@          
          @@@@@ @@@@@@@@@@@@@@@@@@@@@@@@@     @@@@@         
          @@@@@ @@@@@@@@@@@@@@@@@@@@@@@@@     @@@@@         
           @@@@  @@@@@@@@@@@@@@@@@@@@@@@@   @@@@@@          
           @@@@@ @@@@@@@@@@@@@@@@@@@@@@@   @@@@@@           
            @@@@@ @@@@@@@@@@@@@@@@@@@@@@  @@@@@             
             @@@@@@@@@@@@@@@@@@@@@@@@@@                     
              @@@@@@@@@@@@@@@@@@@@@@@                       
                @@@@@@@@@@@@@@@@@@@                         
                                                            
                                                                                                                   
    ~~~~~~~~~~~~ Your mocafe is ready! ~~~~~~~~~~~~~~~
    """
    try:
        import mocafe
        import mocafe.angie
        import mocafe.fenut
        import mocafe.litforms
        import mocafe.expressions
        import mocafe.math

        print(welcome_msg)
    except ModuleNotFoundError as e:

        print("There has been an issue")
        raise e


if __name__ == "__main__":
    main()
