def get_products(user_products):
    '''
    Get list of product ids from a groupby object into competitions format
    #Sample:
    #order_id,products
    #17,1 2
    #34,None
    #137,1 2 3
    '''
    #Get list of product ids from a grouped object
    products = [str(product) for product in set(user_products) if not product == 'None']
    #Check if list is empty, which would mean there are no reordered items; if so, replace by a single None.
    if not products:
        products.append('None')
    #concatenate products
    concat_str = ' '.join(products)
    return concat_str


if __name__ == '__main__':
    pass
