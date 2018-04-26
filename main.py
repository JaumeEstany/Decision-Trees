import file_input as fi
import data_clean as dc
import validation_methods as mth
import randomForest as rf


if __name__ == "__main__":

    #INSTRUCCIONS: DECOMENTA EL PATH DE LA BASE DE DADES QUE VULGUIS UTILITZAR
    #ELS RESULTATS S'ANIRAN IMPRIMINT A LA CONSOLA

    # Llegim la base de dades i tractem els possibles valors continus
    path = "resources/agaricus-lepiota.data"
    classif_index = 0
    # path = "resources/adult.data"
    # classif_index = 14
    classif, attrs = fi.load_dataset(path, classif_index)

    if path == "resources/adult.data":
        attrs = dc.processContinuousAttributesAsNormal(attrs)


    classification_possible_values = dc.classificationPossibleValues(classif)
    attributes_possible_values = dc.attributePossibleValues(attrs)


    # RESULTATS AMB HOLDOUT
    print "HOLDOUT: "
    errors = mth.holdout(attrs, classif, 1, attributes_possible_values, classification_possible_values, classif[0], 0.8)
    print errors


    # RESULTATS AMB K-FOLD CROSS VALIDATION
    k = [5, 10, 15, 20, 50]
    for i in k:
        print "K-FOLD AMB K = " + str(i) + ": "
        errors = mth.k_fold(attrs, classif, attributes_possible_values, classification_possible_values, classif[0], i)
        print errors


    # RESULTATS AMB LEAVE ONE OUT CROSS VALIDATION
    print "LEAVE ONE OUT: "
    #errors = mth.leave_one_out(attrs, classif, attributes_possible_values, classification_possible_values, classif[0])
    #print errors


    # RESULTATS AMB BOOTSTRAPING
    print "BOOTSTRAPPING: "
    errors = mth.bootstrapping(attrs, classif, 2, attributes_possible_values, classification_possible_values, classif[0])
    print errors


    # RESULTATS DEL RANDOM FOREST
    arbres = 10
    print "RANDOM FOREST AMB " + str(arbres) + " ARBRES"
    forest = rf.RandomForest(arbres)
    errors = forest.trainAndValidate(attrs, classif, attributes_possible_values, classification_possible_values, classif[0])
    print errors


    # RESULTATS DEL HOLDOUT CLASSIFICANT MISSING VALUES
    print "HOLDOUT CLASSIFICANT MISSING VALUES"
    mth.holdout_with_missing_values(attrs, classif, attributes_possible_values, classification_possible_values, insertMissing=False)


    # RESULTATS DEL HOLDOUT CLASSIFICANT MISSING VALUES INSERTANT INTERROGANTS
    print "HOLDOUT CLASSIFICANT MISSING VALUES INSERTATS DIRECTAMENT"
    mth.holdout_with_missing_values(attrs, classif, attributes_possible_values, classification_possible_values, insertMissing=True)
