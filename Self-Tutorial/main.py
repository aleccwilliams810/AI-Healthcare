########### One overall Main Script

from split_scripts_for_testing import main_pull, main_encode, main_embed, main_kmeans, main_model

if __name__ == '__main__':
    main_pull()
    main_encode()
    main_embed()
    main_kmeans()
    main_model()