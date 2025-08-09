from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import pandas as pd
from typing import List, Optional
import logging

def read_sdf_to_dataframe(sdf_file: str, 
                         properties: Optional[List[str]] = None,
                         compute_fingerprints: bool = False) -> pd.DataFrame:
    """
    Read an SDF file and convert it to a pandas DataFrame.
    
    Args:
        sdf_file (str): Path to the SDF file
        properties (List[str], optional): List of property names to extract from SDF
        compute_fingerprints (bool): Whether to compute Morgan fingerprints
        
    Returns:
        pd.DataFrame: DataFrame containing molecule information
    """
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create a molecule supplier
    sdf_supplier = Chem.SDMolSupplier(sdf_file, removeHs=False)
    
    data = []
    skipped_mols = 0
    
    # Process each molecule
    for i, mol in enumerate(sdf_supplier):
        if mol is not None:
            try:
                data_dict = {}
                
                # Basic molecular properties
                data_dict["SMILES"] = Chem.MolToSmiles(mol)
                data_dict["Molecular_Weight"] = Descriptors.ExactMolWt(mol)
                data_dict["LogP"] = Descriptors.MolLogP(mol)
                data_dict["HBA"] = Descriptors.NumHAcceptors(mol)
                data_dict["HBD"] = Descriptors.NumHDonors(mol)
                data_dict["TPSA"] = Descriptors.TPSA(mol)
                data_dict["Rotatable_Bonds"] = Descriptors.NumRotatableBonds(mol)
                
                # Get all available properties if none specified
                if properties is None:
                    properties = mol.GetPropNames()
                
                # Extract requested properties
                for prop in properties:
                    if mol.HasProp(prop):
                        data_dict[prop] = mol.GetProp(prop)
                
                # Compute Morgan fingerprints if requested
                if compute_fingerprints:
                    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
                    data_dict["Morgan_Fingerprint"] = fingerprint.ToBitString()
                
                data.append(data_dict)
                
            except Exception as e:
                skipped_mols += 1
                logger.warning(f"Error processing molecule {i}: {str(e)}")
                continue
        else:
            skipped_mols += 1
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Log summary
    logger.info(f"Successfully processed {len(data)} molecules")
    if skipped_mols > 0:
        logger.warning(f"Skipped {skipped_mols} molecules due to errors")
    
    return df 